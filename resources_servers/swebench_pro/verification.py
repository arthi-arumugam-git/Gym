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

"""SWE-bench Pro patch verification using a NeMo Gym sandbox.

This module ports the verification contract from:
https://github.com/scaleapi/SWE-bench_Pro-os/blob/ca10a60a5fcae51e6948ffe1485d4153d421e6c5/swe_bench_pro_eval.py

The upstream file is a standalone evaluator rather than an importable harness
library. It owns Modal/local-Docker sandbox creation, host workspaces, CSV and
patch-file loading, thread-pool concurrency, progress reporting, and aggregate
result persistence. Importing it directly would also require its repository
layout (``helper_code``, ``run_scripts``, and ``dockerfiles``) plus the Modal,
Docker, and pandas dependencies. Unlike the upstream SWE-bench package, it does
not expose a ``run_instance``-style seam where a small container shim can be
substituted.

Kept equivalent to the pinned upstream evaluator:

* remove binary hunks from candidate patches;
* restore ``/app`` to the task's ``base_commit`` and apply ``patch.diff``;
* restore Dockerfile ``ENV`` declarations and run ``before_repo_set_cmd``;
* pass the selected test files to the task's ``run_script.sh``;
* run the task's ``parser.py`` to produce ``output.json``; and
* resolve only when every fail-to-pass and pass-to-pass test is reported passed.

Changed for NeMo Gym:

* ``AsyncSandbox`` replaces both Modal and the Docker SDK;
* sandbox lifecycle, concurrency, and request routing belong to the resources
  server rather than this verifier;
* evaluator assets are embedded in prepared JSONL rows from pinned upstream
  revisions instead of read from a checked-out repository at runtime;
* immutable image digests replace case-sensitive Docker Hub tags because some
  OpenSandbox registry mirrors normalize tags;
* list strings are parsed safely instead of using Python ``eval``; and
* per-request outputs and failure details are retained in Gym's log directory.
"""

import ast
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox


WORKSPACE_DIR = "/workspace"
REPOSITORY_DIR = "/app"
PATCH_PATH = f"{WORKSPACE_DIR}/patch.diff"
RUN_SCRIPT_PATH = f"{WORKSPACE_DIR}/run_script.sh"
PARSER_PATH = f"{WORKSPACE_DIR}/parser.py"
ENTRY_SCRIPT_PATH = f"{WORKSPACE_DIR}/entryscript.sh"
STDOUT_PATH = f"{WORKSPACE_DIR}/stdout.log"
STDERR_PATH = f"{WORKSPACE_DIR}/stderr.log"
OUTPUT_PATH = f"{WORKSPACE_DIR}/output.json"


@dataclass(frozen=True)
class VerificationInputs:
    instance_id: str
    base_commit: str
    patch: str
    run_script: str
    parser_script: str
    selected_test_files_to_run: str | list[str]
    fail_to_pass: str | list[str]
    pass_to_pass: str | list[str]
    before_repo_set_cmd: str = ""
    base_dockerfile: str = ""
    instance_dockerfile: str = ""


@dataclass(frozen=True)
class VerificationResult:
    completed: bool
    resolved: bool
    patch_applied: bool
    test_results: dict[str, Any] | None
    error: str | None = None


def parse_string_list(value: str | list[str]) -> list[str]:
    """Parse JSON/Python list strings without executing dataset content."""
    if isinstance(value, list):
        parsed = value
    elif not value.strip():
        parsed = []
    else:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)

    if not isinstance(parsed, (list, tuple)) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Expected a list of strings, got {value!r}")
    return list(parsed)


def strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections, matching the public Pro evaluator."""
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept = [
        section
        for section in sections
        if section.strip()
        and not re.search(r"^Binary files .* differ$", section, re.MULTILINE)
        and not re.search(r"^GIT binary patch$", section, re.MULTILINE)
    ]
    return "".join(kept)


def _dockerfile_environment_exports(*dockerfiles: str) -> str:
    exports = []
    for dockerfile in dockerfiles:
        for raw_line in dockerfile.splitlines():
            line = raw_line.strip()
            if line.startswith("ENV "):
                exports.append(line.replace("ENV ", "export ", 1))
    return "\n".join(exports)


def build_entry_script(inputs: VerificationInputs) -> str:
    """Build the in-sandbox evaluator script from trusted benchmark metadata."""
    selected_tests = ",".join(parse_string_list(inputs.selected_test_files_to_run))
    setup_lines = [line.strip() for line in inputs.before_repo_set_cmd.splitlines() if line.strip()]
    setup_command = setup_lines[-1] if setup_lines else ":"
    environment_exports = _dockerfile_environment_exports(inputs.base_dockerfile, inputs.instance_dockerfile)

    return f"""#!/bin/bash
set +e
{environment_exports}
cd {shlex.quote(REPOSITORY_DIR)} || exit 1
git reset --hard {shlex.quote(inputs.base_commit)}
git checkout {shlex.quote(inputs.base_commit)}
git apply -v {shlex.quote(PATCH_PATH)}
PATCH_APPLY_STATUS=$?
{setup_command}
bash {shlex.quote(RUN_SCRIPT_PATH)} {shlex.quote(selected_tests)} > {shlex.quote(STDOUT_PATH)} 2> {shlex.quote(STDERR_PATH)}
python {shlex.quote(PARSER_PATH)} {shlex.quote(STDOUT_PATH)} {shlex.quote(STDERR_PATH)} {shlex.quote(OUTPUT_PATH)}
PARSER_STATUS=$?
printf '%s\\n' "$PATCH_APPLY_STATUS" > {shlex.quote(WORKSPACE_DIR + "/patch_apply_status")}
exit "$PARSER_STATUS"
"""


def required_tests_passed(
    test_results: dict[str, Any], fail_to_pass: str | list[str], pass_to_pass: str | list[str]
) -> bool:
    required = set(parse_string_list(fail_to_pass)) | set(parse_string_list(pass_to_pass))
    passed = {
        test["name"]
        for test in test_results.get("tests", [])
        if isinstance(test, dict) and test.get("status") == "PASSED" and isinstance(test.get("name"), str)
    }
    return bool(required) and required <= passed


async def run_verification(
    sandbox: AsyncSandbox,
    inputs: VerificationInputs,
    log_dir: Path,
    timeout_s: int | None,
) -> VerificationResult:
    """Apply one patch, execute Pro's task scripts, and grade their JSON output."""
    cleaned_patch = strip_binary_hunks(inputs.patch)
    entry_script = build_entry_script(inputs)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "patch.diff").write_text(cleaned_patch, encoding="utf-8")
    (log_dir / "eval.sh").write_text(entry_script, encoding="utf-8")

    await sandbox.exec(f"chmod +x {shlex.quote(RUN_SCRIPT_PATH)} {shlex.quote(ENTRY_SCRIPT_PATH)}")
    execution = None
    execution_error = None
    try:
        execution = await sandbox.exec(
            f"bash {shlex.quote(ENTRY_SCRIPT_PATH)}",
            timeout_s=timeout_s,
        )
    except Exception as exc:
        execution_error = exc

    async def capture(remote_path: str, local_name: str) -> str:
        try:
            result = await sandbox.exec(f"cat {shlex.quote(remote_path)}")
            contents = result.stdout or ""
        except Exception as exc:
            contents = f"Failed to read {remote_path}: {exc}\n"
        (log_dir / local_name).write_text(contents, encoding="utf-8")
        return contents

    await capture(STDOUT_PATH, "test_stdout.log")
    await capture(STDERR_PATH, "test_stderr.log")
    patch_status = await capture(f"{WORKSPACE_DIR}/patch_apply_status", "patch_apply_status")
    output_text = await capture(OUTPUT_PATH, "output.json")
    patch_applied = patch_status.strip() == "0"
    (log_dir / "execution_stdout.log").write_text(
        (execution.stdout if execution is not None else "") or "", encoding="utf-8"
    )
    (log_dir / "execution_stderr.log").write_text(
        (execution.stderr if execution is not None else "") or "", encoding="utf-8"
    )

    if execution_error is not None:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            error=f"Evaluation execution failed: {execution_error}",
        )

    try:
        test_results = json.loads(output_text)
    except json.JSONDecodeError as exc:
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            error=f"Parser produced invalid JSON: {exc}",
        )
    if not isinstance(test_results, dict):
        return VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=patch_applied,
            test_results=None,
            error="Parser output must be a JSON object",
        )

    resolved = patch_applied and required_tests_passed(test_results, inputs.fail_to_pass, inputs.pass_to_pass)
    return VerificationResult(
        completed=execution.return_code == 0,
        resolved=resolved,
        patch_applied=patch_applied,
        test_results=test_results,
        error=None if execution.return_code == 0 else f"Evaluation script exited with {execution.return_code}",
    )
