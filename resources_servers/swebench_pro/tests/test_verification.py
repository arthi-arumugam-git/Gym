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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from resources_servers.swebench_pro.verification import (
    VerificationInputs,
    build_entry_script,
    parse_string_list,
    required_tests_passed,
    run_verification,
    strip_binary_hunks,
)


def make_inputs(**overrides) -> VerificationInputs:
    values = {
        "instance_id": "instance_test",
        "base_commit": "abc123",
        "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
        "run_script": "#!/bin/bash\nexit 0\n",
        "parser_script": "import json\n",
        "selected_test_files_to_run": '["a.py", "b.py"]',
        "fail_to_pass": '["test_new"]',
        "pass_to_pass": '["test_old"]',
    }
    values.update(overrides)
    return VerificationInputs(**values)


def test_parse_string_list_accepts_json_python_and_lists() -> None:
    assert parse_string_list('["a", "b"]') == ["a", "b"]
    assert parse_string_list("['a', 'b']") == ["a", "b"]
    assert parse_string_list(["a"]) == ["a"]


def test_parse_string_list_rejects_non_string_items() -> None:
    with pytest.raises(ValueError):
        parse_string_list("[1]")


def test_strip_binary_hunks_preserves_text_diffs() -> None:
    patch = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
        "diff --git a/image.png b/image.png\nGIT binary patch\nliteral 1\nA\n"
    )
    assert strip_binary_hunks(patch) == "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"


def test_build_entry_script_matches_pro_contract() -> None:
    inputs = make_inputs(
        before_repo_set_cmd="ignored setup line\nnpm install",
        base_dockerfile="ENV FOO=bar\n",
    )
    script = build_entry_script(inputs)
    assert "git reset --hard abc123" in script
    assert "git apply -v /workspace/patch.diff" in script
    assert "npm install" in script
    assert "bash /workspace/run_script.sh a.py,b.py" in script
    assert "export FOO=bar" in script


def test_required_tests_passed_requires_all_named_tests() -> None:
    output = {
        "tests": [
            {"name": "test_new", "status": "PASSED"},
            {"name": "test_old", "status": "PASSED"},
        ]
    }
    assert required_tests_passed(output, '["test_new"]', '["test_old"]')
    assert not required_tests_passed(output, '["missing"]', '["test_old"]')
    assert not required_tests_passed(output, "[]", "[]")


@pytest.mark.asyncio
async def test_run_verification_returns_resolved_result(tmp_path) -> None:
    output = {
        "tests": [
            {"name": "test_new", "status": "PASSED"},
            {"name": "test_old", "status": "PASSED"},
        ]
    }
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=0, stdout="run", stderr=""),
                SimpleNamespace(return_code=0, stdout="test stdout", stderr=""),
                SimpleNamespace(return_code=0, stdout="test stderr", stderr=""),
                SimpleNamespace(return_code=0, stdout="0\n", stderr=""),
                SimpleNamespace(return_code=0, stdout=json.dumps(output), stderr=""),
            ]
        )
    )

    result = await run_verification(sandbox, make_inputs(), tmp_path, timeout_s=30)

    assert result.completed
    assert result.resolved
    assert result.patch_applied
    assert json.loads((tmp_path / "output.json").read_text()) == output


@pytest.mark.asyncio
async def test_run_verification_rejects_malformed_parser_output(tmp_path) -> None:
    sandbox = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=1, stdout="", stderr="failed"),
                SimpleNamespace(return_code=0, stdout="", stderr=""),
                SimpleNamespace(return_code=0, stdout="failed", stderr=""),
                SimpleNamespace(return_code=0, stdout="1\n", stderr=""),
                SimpleNamespace(return_code=0, stdout="not-json", stderr=""),
            ]
        )
    )

    result = await run_verification(sandbox, make_inputs(), tmp_path, timeout_s=30)

    assert not result.completed
    assert not result.resolved
    assert "invalid JSON" in result.error
