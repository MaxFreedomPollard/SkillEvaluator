# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for check_error_recovery unrecovered-failure scoring (issue #149).

An observed command failure that is never followed by a successful similar
command must NOT be scored as first-attempt clean.
"""

from __future__ import annotations

import pytest

from skillevaluator.tier3.eval_core.checks import check_error_recovery
from skillevaluator.tier3.harbor.report import _collect_fail_reasons
from skillevaluator.tier3.harbor.templates import eval as eval_template


@pytest.fixture(params=[check_error_recovery, eval_template.check_error_recovery], ids=["host", "harbor-verifier"])
def checker(request):
    return request.param


def _call(cmd: str, observation: str) -> dict:
    return {"action": "bash", "action_input": {"command": cmd}, "observation": observation}


def test_failure_with_no_retry_is_not_first_attempt_clean(checker):
    """A failure that is never retried must not score 1.0 / passed-clean."""
    result = checker([_call("pytest tests/test_api.py -x", "ERROR: test_create_user failed\nexit code 1")])
    assert result["first_attempt_clean"] is False
    assert result["score"] == 0.0
    assert len(result["unrecovered_failures"]) == 1
    assert result["unrecovered_failures"][0]["failed_cmd"] == "pytest tests/test_api.py -x"
    assert result["unrecovered_failures"][0]["recovered"] is False
    assert "1 unrecovered failure" in result["reason"]
    assert "pytest tests/test_api.py -x" in result["reason"]


def test_failed_retries_are_not_first_attempt_clean(checker):
    """Repeated failures of a similar command are all recorded as unrecovered."""
    result = checker(
        [
            _call("pytest tests/test_api.py -x", "ERROR: test_create_user failed\nexit code 1"),
            _call("pytest tests/test_api.py -x -v", "ERROR: test_create_user failed\nexit code 1"),
        ]
    )
    assert result["first_attempt_clean"] is False
    assert result["score"] == 0.0
    assert len(result["failures"]) == 2
    assert len(result["unrecovered_failures"]) == 2
    assert result["corrections"] == []


def test_recovered_failure_behavior_unchanged(checker):
    """A failure followed by a successful similar command keeps the old scoring."""
    result = checker(
        [
            _call("pytest tests/test_api.py", "pytest: error: unrecognized arguments: --bogus\nexit code 2"),
            _call("pytest tests/test_api.py -q", "12 passed"),
        ]
    )
    assert result["first_attempt_clean"] is False
    assert result["unrecovered_failures"] == []
    assert len(result["corrections"]) == 1
    assert result["corrections"][0]["failed_cmd"] == "pytest tests/test_api.py"
    assert result["corrections"][0]["retry_cmd"] == "pytest tests/test_api.py -q"
    # Unchanged pre-existing scoring: one corrected agent fault -> 0.9, passed.
    assert result["score"] == 0.9
    assert result["passed"] is True


def test_unrecovered_failure_after_earlier_success(checker):
    """An unrecovered failure later in the log still breaks first_attempt_clean."""
    result = checker(
        [
            _call("ls -la", "total 4\ndrwxr-xr-x  2 root root 4096 ."),
            _call("make build", "gcc: error: missing.c: No such file or directory"),
        ]
    )
    assert result["first_attempt_clean"] is False
    assert result["score"] == 0.0
    assert len(result["failures"]) == 1
    assert len(result["unrecovered_failures"]) == 1
    assert result["unrecovered_failures"][0]["fault"] == "skill"
    assert "make build" in result["reason"]


def test_unrecovered_failure_after_earlier_recovery(checker):
    result = checker(
        [
            _call("pytest tests/test_api.py", "pytest: error: unrecognized arguments: --bogus\nexit code 2"),
            _call("pytest tests/test_api.py -q", "12 passed"),
            _call("make build", "gcc: error: missing.c: No such file or directory"),
        ]
    )
    assert len(result["corrections"]) == 1
    assert result["first_attempt_clean"] is False
    assert [f["failed_cmd"] for f in result["unrecovered_failures"]] == ["make build"]
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["skill_faults"] == 1
    assert result["agent_faults"] == 1


@pytest.mark.parametrize(
    "observation",
    [
        "Process exited with code 0\nFinal output:\n0 errors",
        "exit_code=0\nERROR: example diagnostic",
        "exit code: 0\nTraceback (most recent call last):",
        "0 errors",
        "No errors found",
        "error handling example",
        "timeout configured to 30 seconds",
        "try:\n    open(p)\nexcept FileNotFoundError:\n    pass",
        "server:\n  timeout: 30",
    ],
)
def test_successful_output_is_not_failure(checker, observation):
    result = checker([_call("make build", observation)])
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize(
    "observation",
    [
        "Process exited with code 7\nFinal output:\nAll done",
        "exit_code=127",
        "exit code: 42",
        "exit code -9",
        "Timeout: command exceeded 30 seconds",
        "TimeoutError",
        "FileNotFoundError",
        "bash: foo: not found",
        "ConnectionRefused",
        "timeout",
        "ValueError: invalid value",
        "Traceback (most recent call last):",
        "status=failed",
        "command not found",
    ],
)
def test_failure_detection_agrees(checker, observation):
    result = checker([_call("run.py", observation)])
    assert result["score"] == 0.0
    assert result["first_attempt_clean"] is False
    assert len(result["unrecovered_failures"]) == 1
    assert result["skill_faults"] + result["agent_faults"] == 1


def test_unrecovered_skill_fault_report_reason(checker):
    result = checker([_call("make build", "gcc: error: missing.c: No such file or directory")])
    assert result["passed"] is False
    assert result["skill_faults"] == 1
    assert result["agent_faults"] == 0
    reasons = _collect_fail_reasons("skill_execution", [{"detail": {"error_recovery": result}}])
    assert result["reason"] in reasons
    assert "make build" in result["reason"]


def test_recovered_skill_fault_counted_once(checker):
    result = checker(
        [
            _call("make build", "No such file or directory"),
            _call("make build", "done"),
        ]
    )
    assert result["skill_faults"] == 1
    assert result["score"] == 0.75
    assert result["passed"] is False
    assert result["unrecovered_failures"] == []


@pytest.mark.parametrize(
    "observation",
    [
        "ERROR: test failed\nexit code 1",
        "Traceback (most recent call last):\nValueError: bad input",
    ],
)
def test_unrecovered_agent_fault_report_reason(checker, observation):
    result = checker([_call("pytest", observation)])
    assert result["score"] == 0.0
    assert result["passed"] is False
    assert result["agent_faults"] == 1
    assert result["skill_faults"] == 0
    reasons = _collect_fail_reasons("skill_execution", [{"detail": {"error_recovery": result}}])
    assert result["reason"] in reasons


@pytest.mark.parametrize(
    "observation",
    [
        "def failure(stage, message, exit_code=1):",
        "    translate unexpected failures to exit code 2",
        "try:\n    open(p)\nexcept Exception:\n    pass",
        'print("Error: expected <spec_url>")',
        'f"{self.prog}: error: {message}"',
    ],
)
def test_displayed_source_is_not_execution_failure(checker, observation):
    result = checker([_call("cat loader.py", observation)])
    assert result["score"] == 1.0
    assert result["passed"] is True
    assert result["failures"] == []


@pytest.mark.parametrize("inspections", [4, 5, 6, 8])
def test_recovery_after_intervening_inspections(checker, inspections):
    calls = [_call("pytest -q", "Process exited with code 1\nFinal output:\nERROR: test failed")]
    calls.extend(
        _call(f"sed -n '1,20p' file{i}.py", "Process exited with code 0\nFinal output:\nsource")
        for i in range(inspections)
    )
    calls.append({"action": "apply_patch", "action_input": {}, "observation": "Done"})
    calls.append(_call("pytest -q", "Process exited with code 0\nFinal output:\n12 passed"))
    result = checker(calls)
    assert result["passed"] is True
    assert result["score"] == 0.9
    assert len(result["corrections"]) == 1
    assert result["corrections"][0]["steps_to_fix"] == inspections + 2
    assert result["unrecovered_failures"] == []


@pytest.mark.parametrize(
    "command",
    [
        "rg TODO src",
        "grep needle file.txt",
        "test -f out.json",
        "[ -e out.json ]",
        "diff -q a b",
        "cmp -s a b",
        "/usr/bin/grep needle file.txt",
    ],
)
@pytest.mark.parametrize(
    "observation", ["Exit code 1", "status=completed\nexit_code=1", "Process exited with code 1\nFinal output:\n"]
)
def test_quiet_negative_probe_is_not_failure(checker, command, observation):
    result = checker([_call(command, observation)])
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["failures"] == []


@pytest.mark.parametrize(
    "command,observation",
    [
        ("pytest -q", "Exit code 1"),
        ("python run.py", "Exit code 1"),
        ("rg TODO src", "Exit code 2"),
        ("rg TODO src", "Exit code 1\nrg: src: Permission denied"),
        ("test -f out.json", "Exit code 1\ntest: invalid argument"),
        ("rg TODO src; python run.py", "Exit code 1"),
        ("rg TODO src && python run.py", "Exit code 1"),
        ("rg TODO src\npython run.py", "Exit code 1"),
        ("rg TODO src | python run.py", "Exit code 1"),
        ("git status", "Exit code 128\nfatal: not a git repository"),
    ],
)
def test_quiet_probe_exception_does_not_hide_real_failures(checker, command, observation):
    result = checker([_call(command, observation)])
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert len(result["unrecovered_failures"]) == 1


@pytest.mark.parametrize(
    "status", ["Exit code 2", "exit_code=2", "[exit_code=2]", "[exit_code] 2", "Process exited with code 2"]
)
def test_whole_status_lines_remain_failures(checker, status):
    result = checker([_call("python run.py", status)])
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_negative_probe_is_not_successful_recovery(checker):
    result = checker([_call("rg TODO src", "Exit code 2\nrg: Permission denied"), _call("rg TODO src", "Exit code 1")])
    assert result["passed"] is False
    assert result["corrections"] == []
    assert len(result["failures"]) == 1
    assert len(result["unrecovered_failures"]) == 1
