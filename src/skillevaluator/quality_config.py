# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
_TOP_LEVEL_KEYS = frozenset({"expected_headers"})
_EXPECTED_HEADER_KEYS = frozenset({"heading", "alternatives", "points", "message", "suggestion"})


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


def _normalize_heading(value: object, source: str, location: str) -> str:
    heading_match = _ATX_HEADING_RE.fullmatch(value) if isinstance(value, str) else None
    heading_title = heading_match.group(2).strip() if heading_match else ""
    if heading_match is None or not heading_title:
        raise QualityConfigError(f"{source}: {location} must be a single ATX Markdown heading")
    return f"{heading_match.group(1)} {heading_title}"


def _reject_unknown_keys(mapping: dict, allowed: frozenset[str], source: str, location: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise QualityConfigError(
            f"{source}: unknown key(s) in {location}: {', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})"
        )


def _yaml_error_detail(exc: yaml.YAMLError) -> str:
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None)
    if mark is None or problem is None:
        return str(exc)
    return f"line {mark.line + 1}, column {mark.column + 1}: {problem}"


def load_quality_config(path: Path | None = None) -> QualityConfig:
    source = str(path) if path else "packaged quality-default.yaml"
    try:
        text = path.read_text(encoding="utf-8") if path else DEFAULT_QUALITY_CONFIG.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise QualityConfigError(f"{source}: file is not valid UTF-8") from exc
    except OSError as exc:
        raise QualityConfigError(f"{source}: cannot read file: {exc.strerror or exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise QualityConfigError(f"{source}: invalid YAML: {_yaml_error_detail(exc)}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("expected_headers"), list):
        raise QualityConfigError(f"{source}: expected_headers must be a list")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, source, "top level")

    expected_headers: list[ExpectedHeader] = []
    seen_headings: dict[str, str] = {}
    for index, rule in enumerate(raw["expected_headers"]):
        location = f"expected_headers[{index}]"
        if not isinstance(rule, dict):
            raise QualityConfigError(f"{source}: {location} must be a mapping")
        _reject_unknown_keys(rule, _EXPECTED_HEADER_KEYS, source, location)

        heading = _normalize_heading(rule.get("heading"), source, f"{location}.heading")
        raw_alternatives = rule.get("alternatives", [])
        if not isinstance(raw_alternatives, list):
            raise QualityConfigError(f"{source}: {location}.alternatives must be a list")
        alternatives = [
            _normalize_heading(value, source, f"{location}.alternatives[{alternative_index}]")
            for alternative_index, value in enumerate(raw_alternatives)
        ]
        points = rule.get("points", 5.0)
        if isinstance(points, bool) or not isinstance(points, (int, float)) or not isfinite(points) or points < 0:
            raise QualityConfigError(f"{source}: {location}.points must be a non-negative number")
        message = rule.get("message")
        suggestion = rule.get("suggestion")
        for field, value in (("message", message), ("suggestion", suggestion)):
            if not isinstance(value, str) or not value.strip():
                raise QualityConfigError(f"{source}: {location}.{field} must be a non-empty string")

        for accepted in (heading, *alternatives):
            if accepted in seen_headings:
                raise QualityConfigError(
                    f"{source}: duplicate heading '{accepted}' in {location} (already listed in {seen_headings[accepted]})"
                )
            seen_headings[accepted] = location

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
