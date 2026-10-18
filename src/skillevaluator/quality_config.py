"""Configuration for deterministic quality scoring."""
from __future__ import annotations

import re
from importlib.resources import files
from math import isfinite
from pathlib import Path
from typing import TypedDict

import yaml

DEFAULT_QUALITY_CONFIG = files("skillevaluator.config").joinpath("quality-default.yaml")
_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$")


class QualityConfigError(ValueError):
    """Raised when a quality configuration is invalid."""


class ExpectedHeader(TypedDict):
    """A Markdown heading expected by the quality validator."""

    heading: str
    alternatives: list[str]
    points: float
    message: str
    suggestion: str


class QualityConfig(TypedDict):
    """Validated deterministic quality configuration."""

    expected_headers: list[ExpectedHeader]


def _normalize_heading(value: object, location: str) -> str:
    heading_match = _ATX_HEADING_RE.fullmatch(value) if isinstance(value, str) else None
    heading_title = heading_match.group(2).strip() if heading_match else ""
    if heading_match is None or not heading_title:
        raise QualityConfigError(f"quality config: {location} must be a single ATX Markdown heading")
    return f"{heading_match.group(1)} {heading_title}"


def load_quality_config(path: Path | None = None) -> QualityConfig:
    source = path.read_text(encoding="utf-8") if path else DEFAULT_QUALITY_CONFIG.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise QualityConfigError(f"{path or 'packaged quality-default.yaml'}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("expected_headers"), list):
        raise QualityConfigError("quality config: expected_headers must be a list")

    expected_headers: list[ExpectedHeader] = []
    for index, rule in enumerate(raw["expected_headers"]):
        if not isinstance(rule, dict):
            raise QualityConfigError(f"quality config: expected_headers[{index}] must be a mapping")

        heading = _normalize_heading(rule.get("heading"), f"expected_headers[{index}].heading")
        raw_alternatives = rule.get("alternatives", [])
        if not isinstance(raw_alternatives, list):
            raise QualityConfigError(f"quality config: expected_headers[{index}].alternatives must be a list")
        alternatives = [
            _normalize_heading(value, f"expected_headers[{index}].alternatives[{alternative_index}]")
            for alternative_index, value in enumerate(raw_alternatives)
        ]
        points = rule.get("points", 5.0)
        if isinstance(points, bool) or not isinstance(points, (int, float)) or not isfinite(points) or points < 0:
            raise QualityConfigError(
                f"quality config: expected_headers[{index}].points must be a non-negative number"
            )
        message = rule.get("message")
        suggestion = rule.get("suggestion")
        for field, value in (("message", message), ("suggestion", suggestion)):
            if not isinstance(value, str) or not value.strip():
                raise QualityConfigError(
                    f"quality config: expected_headers[{index}].{field} must be a non-empty string"
                )
        expected_headers.append(
            {
                "heading": heading,
                "alternatives": alternatives,
                "points": float(points),
                "message": message,
                "suggestion": suggestion,
            }
        )

    return {"expected_headers": expected_headers}
