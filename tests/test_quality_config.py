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
        'expected_headers:\n  - heading: "## Limitations"\n    suggestion: Add it\n',
        'expected_headers:\n  - heading: "## Limitations"\n    message: Missing\n',
    ],
)
def test_quality_config_rejects_invalid_expected_headers(tmp_path: Path, config: str) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(QualityConfigError, match="expected_headers"):
        load_quality_config(config_path)
