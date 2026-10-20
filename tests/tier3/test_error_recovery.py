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
