# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

import requests


def _extract_failed_tests(summary_short: str) -> list[tuple[str, str]]:
    """
    Return a list of tuples (<test node id>, <error message from short summary>).
    """
    failed_tests = []
    for line in summary_short.splitlines():
        if not line.startswith("FAILED "):
            continue
        # Skip subprocess failures created by `run_test_using_subprocess`
        if " - Failed: (subprocess)" in line:
            continue
        failure_line = line[len("FAILED ") :].strip()
        test_node, sep, error_message = failure_line.partition(" - ")
        failed_tests.append((test_node.strip(), error_message.strip()))

    return failed_tests


def _extract_failure_lines(failures_line: str | None) -> list[str]:
    if not failures_line:
        return []

    failure_lines = []
    for raw_line in failures_line.splitlines():
        raw_line = raw_line.strip()
        if (
            not raw_line
            or raw_line.startswith("=")
            or raw_line.startswith("_")
            or raw_line.lower().startswith("short test summary")
        ):
            continue
        if ": " not in raw_line:
            continue
        failure_lines.append(raw_line)

    return failure_lines


def _derive_model_name(test_node_id: str) -> str | None:
    """
    Given a pytest node id (e.g. tests/models/bart/test_modeling_bart.py::BartModelTest::test_forward),
    extract the model name when it lives under `tests/models`.
    """
    file_path = test_node_id.split("::", maxsplit=1)[0]
    if file_path.startswith("tests/models/"):
        parts = file_path.split("/")
        if len(parts) >= 3:
            return parts[2]
    return None


def _aggregate_failures(failure_entries: list[dict]) -> tuple[dict, dict]:
    by_test: dict[str, dict] = {}
    by_model: dict[str, dict] = {}

    for entry in failure_entries:
        test_name = entry["test_name"]
        model_name = entry["model_name"]
        error_message = entry["error"]
        normalized_test_name = _normalize_test_nodeid(test_name)

        test_info = by_test.setdefault(
            normalized_test_name, {"count": 0, "errors": Counter(), "jobs": set(), "variants": set()}
        )
        test_info["count"] += 1
        test_info["errors"][error_message] += 1
        test_info["jobs"].add(entry["job_name"])
        test_info["variants"].add(test_name)

        if model_name:
            model_info = by_model.setdefault(model_name, {"count": 0, "errors": Counter(), "tests": set()})
            model_info["count"] += 1
            model_info["errors"][error_message] += 1
            model_info["tests"].add(test_name)

    # Convert counters and sets to serializable forms
    def _prepare(entries: dict, include_tests: bool = False):
        prepared = {}
        for key, value in entries.items():
            prepared[key] = {
                "count": value["count"],
                "errors": dict(value["errors"].most_common()),
            }
            if include_tests:
                prepared[key]["tests"] = sorted(value["tests"])
            else:
                prepared[key]["jobs"] = sorted(value["jobs"])
                prepared[key]["variants"] = sorted(value["variants"])
        return prepared

    return _prepare(by_test), _prepare(by_model, include_tests=True)


def _format_error_messages(errors: dict[str, int]) -> str:
    return "; ".join(f"{count}× {msg}" for msg, count in errors.items()) or "N/A"


def _format_markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    if not rows:
        return "No data\n"

    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    table_lines = [header_line, separator]
    table_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table_lines) + "\n"


def _normalize_test_nodeid(nodeid: str) -> str:
    """
    Normalizes a pytest node id by removing bracketed parametrization info
    and collapsing suffixes such as `_05_fp16_pad_left` that come from parameter ids.
    """
    base_nodeid = nodeid.split("[", 1)[0]
    parts = base_nodeid.split("::")
    if not parts:
        return base_nodeid
    test_name = parts[-1]
    test_name = re.sub(r"_\d{2,}.*$", "", test_name)
    normalized = "::".join(parts[:-1] + [test_name])
    return normalized


def _collect_metadata(workflow_id: str) -> dict[str, str | None]:
    repo_owner = os.environ.get("CIRCLE_PROJECT_USERNAME")
    repo_name = os.environ.get("CIRCLE_PROJECT_REPONAME")
    repo_slug = "/".join(part for part in [repo_owner, repo_name] if part)
    commit_sha = os.environ.get("CIRCLE_SHA1")
    branch = os.environ.get("CIRCLE_BRANCH")
    pull_request = os.environ.get("CIRCLE_PULL_REQUEST")
    pr_number = os.environ.get("CIRCLE_PR_NUMBER")
    if not pr_number and pull_request and "/" in pull_request:
        pr_number = pull_request.rsplit("/", 1)[-1]
    build_num = os.environ.get("CIRCLE_BUILD_NUM")
    timestamp = os.environ.get("CIRCLE_WORKFLOW_CREATED_AT")
    if not timestamp:
        timestamp = datetime.now(timezone.utc).isoformat()
    commit_short = (commit_sha or "unknown")[:8]
    dataset_subfolder = f"{repo_slug.replace('/', '__') or 'unknown_repo'}/pr-{pr_number or 'none'}/sha-{commit_short}/workflow-{workflow_id}"
    metadata = {
        "workflow_id": workflow_id,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "repository": repo_slug,
        "branch": branch,
        "commit_sha": commit_sha,
        "pull_request": pull_request,
        "pull_request_number": pr_number,
        "build_number": build_num,
        "collected_at": timestamp,
        "dataset_subfolder": dataset_subfolder,
    }
    return metadata


def process_circleci_workflow(
    workflow_id: str,
    output_dir: str = "outputs",
    request_get: Callable = requests.get,
):
    print(f"[collection_job] Processing CircleCI workflow {workflow_id}")
    response = request_get(
        f"https://circleci.com/api/v2/workflow/{workflow_id}/job",
        headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")},
    )
    jobs = response.json()["items"]
    print(f"[collection_job] Found {len(jobs)} jobs in workflow.")

    os.makedirs(output_dir, exist_ok=True)

    workflow_summary = {}
    failure_entries: list[dict] = []
    # for each job, download artifacts
    for job in jobs:
        project_slug = job["project_slug"]
        if job["name"].startswith(("tests_", "examples_", "pipelines_")):
            print(f"[collection_job] Fetching artifacts for job {job['name']} (#{job['job_number']})")
            url = f"https://circleci.com/api/v2/project/{project_slug}/{job['job_number']}/artifacts"
            r = request_get(url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
            job_artifacts = r.json()["items"]
            print(f"[collection_job] Retrieved {len(job_artifacts)} artifacts for {job['name']}.")

            job_output_dir = os.path.join(output_dir, job["name"])
            os.makedirs(job_output_dir, exist_ok=True)

            job_test_summaries = {}
            job_failure_lines = {}
            for artifact in job_artifacts:
                if artifact["path"].startswith("reports/") and artifact["path"].endswith("/summary_short.txt"):
                    node_index = artifact["node_index"]
                    artifact_url = artifact["url"]
                    r = request_get(artifact_url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    test_summary = r.text
                    job_test_summaries[node_index] = test_summary
                elif artifact["path"].startswith("reports/") and artifact["path"].endswith("/failures_line.txt"):
                    node_index = artifact["node_index"]
                    artifact_url = artifact["url"]
                    r = request_get(artifact_url, headers={"Circle-Token": os.environ.get("CIRCLE_TOKEN", "")})
                    job_failure_lines[node_index] = r.text

            summary = {}
            for node_index, node_test_summary in job_test_summaries.items():
                for line in node_test_summary.splitlines():
                    if line.startswith("PASSED "):
                        test = line[len("PASSED ") :]
                        summary[test] = "passed"
                    elif line.startswith("FAILED "):
                        test = line[len("FAILED ") :].split()[0]
                        summary[test] = "failed"
            # failed before passed
            summary = dict(sorted(summary.items(), key=lambda x: (x[1], x[0])))
            workflow_summary[job["name"]] = summary
            print(f"[collection_job] Recorded {len(summary)} test rows for {job['name']}.")

            # collected version
            with open(os.path.join(job_output_dir, "test_summary.json"), "w") as fp:
                json.dump(summary, fp, indent=4)

            # Collect failure details per node for this job
            for node_index, summary_short in job_test_summaries.items():
                failed_tests = _extract_failed_tests(summary_short)
                failure_lines = _extract_failure_lines(job_failure_lines.get(node_index))
                for idx, (test_name, short_error) in enumerate(failed_tests):
                    full_error = failure_lines[idx] if idx < len(failure_lines) else short_error
                    failure_entries.append(
                        {
                            "job_name": job["name"],
                            "node_index": node_index,
                            "test_name": test_name,
                            "short_error": short_error,
                            "error": full_error,
                            "model_name": _derive_model_name(test_name),
                        }
                    )
            if job_test_summaries:
                failures_in_job = sum(1 for status in summary.values() if status == "failed")
                print(f"[collection_job] Aggregated {failures_in_job} failures for {job['name']}.")

    new_workflow_summary = {}
    for job_name, job_summary in workflow_summary.items():
        for test, status in job_summary.items():
            if test not in new_workflow_summary:
                new_workflow_summary[test] = {}
            new_workflow_summary[test][job_name] = status

    for test, result in new_workflow_summary.items():
        new_workflow_summary[test] = dict(sorted(result.items()))
    new_workflow_summary = dict(sorted(new_workflow_summary.items()))

    with open(os.path.join(output_dir, "test_summary.json"), "w") as fp:
        json.dump(new_workflow_summary, fp, indent=4)

    failures_by_test, failures_by_model = _aggregate_failures(failure_entries)
    failure_summary = {
        "failures": failure_entries,
        "by_test": failures_by_test,
        "by_model": failures_by_model,
    }
    print(f"[collection_job] Total failing entries collected: {len(failure_entries)}.")

    with open(os.path.join(output_dir, "failure_summary.json"), "w") as fp:
        json.dump(failure_summary, fp, indent=4)

    markdown_buffer = ["# Failure summary\n"]
    if failure_entries:
        markdown_buffer.append("## By test\n")
        test_rows = []
        for test_name, info in sorted(failures_by_test.items(), key=lambda x: x[1]["count"], reverse=True):
            test_rows.append(
                [
                    test_name,
                    str(info["count"]),
                    _format_error_messages(info["errors"]),
                ]
            )
        markdown_buffer.append(_format_markdown_table(test_rows, ["Test", "Failures", "Full error(s)"]))

        markdown_buffer.append("## By model\n")
        model_rows = []
        for model_name, info in sorted(failures_by_model.items(), key=lambda x: x[1]["count"], reverse=True):
            model_rows.append(
                [
                    model_name,
                    str(info["count"]),
                    _format_error_messages(info["errors"]),
                ]
            )
        markdown_buffer.append(_format_markdown_table(model_rows, ["Model", "Failures", "Full error(s)"]))
    else:
        markdown_buffer.append("No failures were reported.\n")

    markdown_text = "\n".join(markdown_buffer)
    with open(os.path.join(output_dir, "failure_summary.md"), "w") as fp:
        fp.write(markdown_text)

    metadata = _collect_metadata(workflow_id)
    aggregate_payload = {
        "metadata": metadata,
        "jobs": workflow_summary,
        "tests": new_workflow_summary,
        "failures": failure_entries,
        "failures_by_test": failures_by_test,
        "failures_by_model": failures_by_model,
    }
    with open(os.path.join(output_dir, "collection_summary.json"), "w") as fp:
        json.dump(aggregate_payload, fp, indent=4)
    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(metadata, fp, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_id", type=str, required=True)
    args = parser.parse_args()
    workflow_id = args.workflow_id
    process_circleci_workflow(workflow_id)


if __name__ == "__main__":
    main()
