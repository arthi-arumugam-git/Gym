# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

from benchmarks.swebench.pro.prepare import UPSTREAM_COMMIT, enrich_row, prepare


def make_upstream(root: Path, instance_id: str) -> None:
    assets = {
        f"run_scripts/{instance_id}/run_script.sh": "#!/bin/bash\n",
        f"run_scripts/{instance_id}/parser.py": "print('parser')\n",
        f"dockerfiles/base_dockerfile/{instance_id}/Dockerfile": "FROM base\n",
        f"dockerfiles/instance_dockerfile/{instance_id}/Dockerfile": "FROM instance\n",
    }
    for relative_path, contents in assets.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def dataset_row() -> dict:
    return {
        "repo": "example/repo",
        "instance_id": "instance_example",
        "base_commit": "abc123",
        "patch": "patch",
        "test_patch": "",
        "problem_statement": "Fix it",
        "fail_to_pass": '["new_test"]',
        "pass_to_pass": '["old_test"]',
        "before_repo_set_cmd": "",
        "selected_test_files_to_run": '["tests"]',
        "dockerhub_tag": "example-tag",
    }


def test_enrich_row_embeds_pinned_evaluator_assets(tmp_path) -> None:
    make_upstream(tmp_path, "instance_example")

    row = enrich_row(dataset_row(), tmp_path, "sha256:digest")

    assert row["run_script"] == "#!/bin/bash\n"
    assert row["parser_script"] == "print('parser')\n"
    assert row["base_dockerfile"] == "FROM base\n"
    assert row["image_digest"] == "sha256:digest"
    assert row["evaluator_commit"] == UPSTREAM_COMMIT
    assert row["responses_create_params"]["input"][0]["content"] == "Fix it"


def test_prepare_writes_self_contained_jsonl(tmp_path) -> None:
    make_upstream(tmp_path, "instance_example")
    output = tmp_path / "output.jsonl"

    result = prepare(
        dataset=[dataset_row()],
        upstream_root=tmp_path,
        output_fpath=output,
        image_digest_resolver=lambda _: "sha256:digest",
    )

    assert result == output
    prepared = json.loads(output.read_text(encoding="utf-8"))
    assert prepared["instance_id"] == "instance_example"
    assert prepared["parser_script"] == "print('parser')\n"
