#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team.
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
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

DEFAULT_DATASET_ID = os.environ.get("CIRCLECI_RESULTS_DATASET_ID", "transformers-community/circleci-test-results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload CircleCI collection outputs to the Hub.")
    parser.add_argument("--source-dir", type=str, default="outputs", help="Directory containing summary files.")
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID, help="Target dataset ID to update.")
    return parser.parse_args()


def _load_metadata(source_dir: Path) -> dict:
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json missing in {source_dir}")
    with metadata_path.open() as fp:
        return json.load(fp)


def _collect_files(source_dir: Path, base_dir: str) -> list[CommitOperationAdd]:
    filenames = [
        "collection_summary.json",
        "failure_summary.json",
        "failure_summary.md",
        "test_summary.json",
        "metadata.json",
    ]
    operations = []
    for filename in filenames:
        path = source_dir / filename
        if not path.exists():
            continue
        remote = f"{base_dir}/{filename}"
        operations.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(path)))
    return operations


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    dataset_id = args.dataset_id
    if not dataset_id:
        raise ValueError("Dataset ID is required.")

    token = os.environ.get("HF_TOKEN") or os.environ.get("TRANSFORMERS_HUB_BOT_HF_TOKEN")
    if not token:
        raise RuntimeError("HF token not provided. Set HF_TOKEN or TRANSFORMERS_HUB_BOT_HF_TOKEN.")

    metadata = _load_metadata(source_dir)
    pr_number = metadata.get("pull_request_number") or "none"
    commit_sha = metadata.get("commit_sha") or "unknown"
    commit_short = commit_sha[:12]
    base_dir = f"pr-{pr_number}/sha-{commit_short}"

    operations = _collect_files(source_dir, base_dir)
    if not operations:
        raise RuntimeError(f"No summary files found in {source_dir}.")

    api = HfApi(token=token)
    api.create_repo(repo_id=dataset_id, repo_type="dataset", exist_ok=True, token=token)

    commit_message = f"Update CircleCI artifacts for PR {pr_number} ({commit_short})"
    api.create_commit(
        repo_id=dataset_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
        token=token,
    )
    print(f"Uploaded {len(operations)} files to {dataset_id}:{base_dir}")


if __name__ == "__main__":
    main()
