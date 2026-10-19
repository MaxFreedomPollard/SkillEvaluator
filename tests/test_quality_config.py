# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from skillevaluator.quality_config import QualityConfigError, load_quality_config


def test_packaged_default_quality_config_is_available() -> None:
    config = load_quality_config()

    assert config["expected_headers"] == [
        {
            "heading": "## Prerequisites",
            "alternatives": ["## Requirements"],
            "points": 5.0,
            "message": "No prerequisites/requirements documented",
            "suggestion": "Document dependencies, API keys, or setup needed",
        },
        {
            "heading": "## Limitations",
            "alternatives": [],
            "points": 5.0,
            "message": "No limitations documented",
            "suggestion": "Add '## Limitations' section with known issues/constraints",
        },
        {
            "heading": "## Troubleshooting",
            "alternatives": ["## Common Issues", "## FAQ"],
            "points": 5.0,
            "message": "No troubleshooting section documented",
            "suggestion": "Add '## Troubleshooting' with Error/Cause/Solution patterns",
        },
    ]


def test_quality_config_can_be_overridden(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        'expected_headers:\n  - heading: "## Failure Constellations"\n'
        '    alternatives:\n      - "## Known Failures"\n'
        "    message: No failure constellations documented\n"
        "    suggestion: Add '## Failure Constellations'\n"
        '  - heading: "### Security"\n    points: 8.5\n'
        "    message: No security notes documented\n"
        "    suggestion: Add '### Security'\n",
        encoding="utf-8",
    )

    config = load_quality_config(config_path)

    assert config["expected_headers"] == [
        {
            "heading": "## Failure Constellations",
            "alternatives": ["## Known Failures"],
            "points": 5.0,
            "message": "No failure constellations documented",
            "suggestion": "Add '## Failure Constellations'",
        },
        {
            "heading": "### Security",
            "alternatives": [],
            "points": 8.5,
            "message": "No security notes documented",
            "suggestion": "Add '### Security'",
        },
    ]


@pytest.mark.parametrize(
    "config",
    [
        "expected_headers: not-a-list\n",
        "expected_headers:\n  - heading: Limitations\n",
        'expected_headers:\n  - heading: "##   "\n',
        'expected_headers:\n  - heading: "## Limitations"\n    alternatives: not-a-list\n',
        'expected_headers:\n  - heading: "## Limitations"\n    points: -1\n',
        'expected_headers:\n  - heading: "## Limitations"\n    message: ""\n',
        'expected_headers:\n  - heading: "## Limitations"\n    suggestion:\n',
        'expected_headers:\n  - heading: "## Limitations"\n    message: [Missing]\n',
    ],
)
def test_quality_config_rejects_invalid_expected_headers(tmp_path: Path, config: str) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match="expected_headers"):
        load_quality_config(config_path)


def test_quality_config_generates_message_and_suggestion_when_omitted(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        'expected_headers:\n  - heading: "## Security"\n'
        '  - heading: "## Prerequisites"\n    alternatives:\n      - "## Requirements"\n    points: 5\n'
        '  - heading: "## Limitations"\n    message: No limitations documented\n',
        encoding="utf-8",
    )

    config = load_quality_config(config_path)

    assert [(rule["message"], rule["suggestion"]) for rule in config["expected_headers"]] == [
        ("Missing expected heading: '## Security'", "Add a '## Security' section"),
        (
            "Missing expected heading: '## Prerequisites' or '## Requirements'",
            "Add one of these sections: '## Prerequisites' or '## Requirements'",
        ),
        ("No limitations documented", "Add a '## Limitations' section"),
    ]


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (
            "expected_headers: []\nexpected_header: []\n",
            r"unknown key\(s\) in top level: expected_header",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    point: 10\n'
            "    message: Missing\n    suggestion: Add it\n",
            r"unknown key\(s\) in expected_headers\[0\]: point",
        ),
        (
            'expected_headers:\n  - heading: "## Limitations"\n    message: Missing\n    suggestion: Add it\n'
            '  - heading: "## Limits"\n    alternatives:\n      - "##  Limitations"\n'
            "    message: Missing\n    suggestion: Add it\n",
            r"duplicate heading '## Limitations' in expected_headers\[1\] \(already listed in expected_headers\[0\]\)",
        ),
    ],
)
def test_quality_config_rejects_unknown_keys_and_duplicate_headings(
    tmp_path: Path, config: str, match: str
) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match=match):
        load_quality_config(config_path)


def test_quality_config_errors_name_the_file_and_yaml_position(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("expected_headers: [\n", encoding="utf-8")

    with pytest.raises(QualityConfigError) as exc_info:
        load_quality_config(config_path)

    message = str(exc_info.value)
    assert message.startswith(f"{config_path}: invalid YAML: line 2, column 1: ")
    assert "\n" not in message


def test_quality_config_rejects_non_utf8_file(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text("expected_headers: []\n", encoding="utf-16")

    with pytest.raises(QualityConfigError, match="not valid UTF-8"):
        load_quality_config(config_path)
