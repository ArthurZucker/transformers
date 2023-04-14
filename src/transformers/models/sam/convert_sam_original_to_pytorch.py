# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert SAM checkpoints from the original repository.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    SamConfig,
    SamForImageSegmentation,
    SamImageProcessor,
    SamProcessor,
    SamVisionConfig,
)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def replace_keys(state_dict):
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    return state_dict


def convert_sam_checkpoint(model_name, pytorch_dump_folder, push_to_hub):
    checkpoint_path = hf_hub_download("ybelkada/segment-anything", f"checkpoints/{model_name}.pth")

    if "sam_vit_b" in model_name:
        config = SamConfig()
    elif "sam_vit_l" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )

        config = SamConfig(
            vision_config=vision_config,
        )
    elif "sam_vit_h" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )

        config = SamConfig(
            vision_config=vision_config,
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = replace_keys(state_dict)

    image_processor = SamImageProcessor()

    processor = SamProcessor(image_processor=image_processor)
    hf_model = SamForImageSegmentation(config)

    hf_model.load_state_dict(state_dict)
    hf_model = hf_model.to("cuda")

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    input_points = ((400, 650),)
    input_labels = (1,)

    inputs = processor(images=np.array(raw_image), return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = hf_model(**inputs)
    scores = output["iou_predictions"].squeeze()

    if model_name == "sam_vit_h_4b8939":
        assert scores[-1].item() == 0.579890251159668

        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output["iou_predictions"].squeeze()

        assert scores[-1].item() == 0.9712603092193604

        input_boxes = ((75, 275, 1725, 850),)

        inputs = processor(images=np.array(raw_image), input_boxes=input_boxes, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output["iou_predictions"].squeeze()

        # TODO: verify logits here
        print(scores)

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     mask = mask.cpu().detach()
    #     plt.imshow(np.array(raw_image))
    #     show_mask(mask, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()

    #     plt.savefig(f"temp_{i}.png")

    # hf_model.save_pretrained(pytorch_dump_folder)
    # processor.save_pretrained(pytorch_dump_folder)

    # if push_to_hub:
    #    hf_model.push_to_hub()
    #    processor.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["sam_vit_b_01ec64", "sam_vit_h_4b8939", "sam_vit_l_0b3195"]
    parser.add_argument(
        "--model_name",
        default="sam_vit_h_4b8939",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    convert_sam_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
