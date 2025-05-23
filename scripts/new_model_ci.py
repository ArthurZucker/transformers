import importlib
import json
import os
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from utils.pr_slow_ci_models import get_new_model


def extract_pr_number():
    try:
        msg = os.popen("git log -1 --pretty=%B").read()
    except Exception:
        return ""
    match = re.search(r"#(\d+)", msg)
    return match.group(1) if match else ""


def check_new_model_label(pr_number):
    if pr_number == "":
        return False
    url = f"https://api.github.com/repos/{os.environ.get('CIRCLE_PROJECT_USERNAME')}/{os.environ.get('CIRCLE_PROJECT_REPONAME')}/pulls/{pr_number}"
    data = json.loads(os.popen(f"curl -L -H 'Accept: application/vnd.github+json' {url}").read())
    labels = [label["name"] for label in data.get("labels", [])]
    return "new-model" in labels


def create_small_model(model_name):
    module = importlib.import_module(f"transformers.models.{model_name}.configuration_{model_name}")
    config_classes = [c for c in module.__dict__.values() if isinstance(c, type) and c.__name__.endswith("Config")]
    config = config_classes[0]()

    for attr in ["hidden_size", "d_model", "intermediate_size", "num_hidden_layers", "num_attention_heads"]:
        if hasattr(config, attr):
            setattr(config, attr, max(1, getattr(config, attr) // 2))

    repo_dir = Path(f"{model_name}-tiny")
    repo_dir.mkdir(exist_ok=True)
    config.save_pretrained(repo_dir)

    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(repo_dir)
    return config, model, repo_dir


def push_to_hub(repo_dir, repo_id):
    token = os.environ.get("HF_TOKEN")
    if token:
        api = HfApi()
        api.upload_folder(folder_path=str(repo_dir), repo_id=repo_id, token=token, repo_type="model")


def train_small_model(repo_dir, repo_id):
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1]")

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True)

    dataset = dataset.map(preprocess, batched=True)
    model = AutoModelForCausalLM.from_pretrained(repo_dir)

    args = TrainingArguments(output_dir="output", num_train_epochs=1, per_device_train_batch_size=1)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    model.save_pretrained(repo_dir)
    push_to_hub(repo_dir, repo_id)


def create_integration_test(model_name, repo_dir):
    test_dir = Path("tests") / "models" / model_name
    test_dir.mkdir(parents=True, exist_ok=True)
    test_file = test_dir / f"test_integration_{model_name}.py"
    content = f"""import unittest
from transformers import AutoModelForCausalLM, AutoTokenizer

class {model_name.capitalize()}IntegrationTest(unittest.TestCase):
    def test_generation(self):
        model = AutoModelForCausalLM.from_pretrained('{repo_dir}')
        tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-gpt2')
        inputs = tokenizer('Hello', return_tensors='pt')
        output = model.generate(**inputs, max_new_tokens=1)
        self.assertEqual(output.shape[0], 1)
"""
    test_file.write_text(content)


if __name__ == "__main__":
    model_name = get_new_model(diff_with_last_commit=True)
    if model_name == "":
        print("No new model detected")
        exit(0)

    pr_number = extract_pr_number()
    if not check_new_model_label(pr_number):
        print("PR does not have new-model label, skipping")
        exit(0)

    repo_id = f"{model_name}-tiny"
    config, model, repo_dir = create_small_model(model_name)
    push_to_hub(repo_dir, repo_id)
    train_small_model(repo_dir, repo_id)
    create_integration_test(model_name, repo_dir)
    print("New model pipeline completed")
