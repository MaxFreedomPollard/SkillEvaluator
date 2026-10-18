# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog runs must account for every live supported skill manifest."""

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.constants import SCAN_EXCLUDED_DIRS
from skillevaluator.utils.helpers import find_skills_in_directory

FIXTURE = Path(__file__).parent / "fixtures" / "skills" / "simple"


def _skill(catalog: Path, relative: str, *, valid: bool = True, filename: str = "SKILL.md") -> Path:
    skill = catalog / relative
    if valid:
        shutil.copytree(FIXTURE, skill)
        manifest = skill / "SKILL.md"
        manifest.write_text(
            manifest.read_text(encoding="utf-8").replace("name: simple", f"name: {skill.name}"),
            encoding="utf-8",
        )
        manifest.rename(skill / filename)
    else:
        skill.mkdir(parents=True)
        (skill / filename).write_text("Missing frontmatter.\n", encoding="utf-8")
    return skill


def _validate(target: Path, reports: Path, workers: int = 1):
    return CliRunner().invoke(
        cli,
        [
            "validate",
            str(target),
            "--tiers",
            "1",
            "--no-dedup",
            "--checks",
            "schema",
            "--workers",
            str(workers),
            "-r",
            "json",
            "-o",
            str(reports),
        ],
    )


@pytest.mark.parametrize("workers", [1, 2])
def test_catalog_includes_nested_and_lowercase_invalid_skills(tmp_path: Path, workers: int) -> None:
    catalog = tmp_path / "catalog"
    _skill(catalog, "simple")
    _skill(catalog, "team-skills/team/broken", valid=False)
    _skill(catalog, "lowercase", valid=False, filename="skill.md")
    reports = tmp_path / "reports"

    result = _validate(catalog, reports, workers)

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 2
    assert summary["overall_passed"] is False
    assert {entry["name"] for entry in summary["skills"]} == {
        "simple",
        "team-skills/team/broken",
        "lowercase",
    }


@pytest.mark.parametrize("relative", ["team-skills/team/broken", "lowercase"])
def test_catalog_is_detected_without_an_immediate_canonical_manifest(tmp_path: Path, relative: str) -> None:
    catalog = tmp_path / "catalog"
    filename = "skill.md" if relative == "lowercase" else "SKILL.md"
    _skill(catalog, relative, valid=False, filename=filename)
    reports = tmp_path / "reports"

    result = _validate(catalog, reports)

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["skills"][0]["name"] == relative
    assert summary["overall_passed"] is False


@pytest.mark.parametrize("workers", [1, 2])
def test_nested_skills_with_the_same_name_keep_separate_reports(tmp_path: Path, workers: int) -> None:
    catalog = tmp_path / "catalog"
    _skill(catalog, "team-a/simple")
    _skill(catalog, "team-b/simple", valid=False)
    reports = tmp_path / "reports"

    result = _validate(catalog, reports, workers)

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in summary["skills"]}
    assert set(entries) == {"team-a/simple", "team-b/simple"}
    assert entries["team-a/simple"]["passed"] is True
    assert entries["team-b/simple"]["passed"] is False
    for name, entry in entries.items():
        assert entry["report_dir"] == name
        report = reports / entry["report_dir"] / entry["json_report"]
        payload = json.loads(report.read_text(encoding="utf-8"))
        assert payload["overall_passed"] is entry["passed"]


def test_catalog_ignores_generated_and_excluded_skill_copies(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    live = _skill(catalog, "simple")
    for excluded in SCAN_EXCLUDED_DIRS:
        _skill(catalog, f"{excluded}/broken", valid=False)
        _skill(catalog, f"team-skills/team/{excluded}/broken", valid=False)
    reports = tmp_path / "reports"

    assert find_skills_in_directory(catalog) == [live]
    result = _validate(catalog, reports)

    assert result.exit_code == 0, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["overall_passed"] is True


def test_lowercase_root_manifest_stays_a_single_skill(tmp_path: Path) -> None:
    skill = _skill(tmp_path, "simple", filename="skill.md")
    _skill(skill, "examples/nested", valid=False)
    reports = tmp_path / "reports"

    result = _validate(skill, reports)

    assert result.exit_code == 1, result.output
    assert not (reports / "catalog-summary.json").exists()
    assert list(reports.glob("*.json"))


def test_discovery_deduplicates_manifest_variants_and_keeps_compatibility_alias(tmp_path: Path) -> None:
    skill = _skill(tmp_path, "simple")
    (skill / "skill.md").write_text("Missing frontmatter.\n", encoding="utf-8")
    (skill / "AGENTS.md").write_text("# Instructions\n", encoding="utf-8")
    (skill / "CLAUDE.md").symlink_to("AGENTS.md")
    (skill / "other-link").symlink_to("missing")

    assert find_skills_in_directory(tmp_path) == [skill]


def test_discovery_does_not_follow_directory_links(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    live = _skill(catalog, "simple")
    external = _skill(tmp_path, "external", valid=False)
    (catalog / "linked").symlink_to(external, target_is_directory=True)

    assert find_skills_in_directory(catalog) == [live]


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_discovery_rejects_unsafe_manifest(tmp_path: Path, kind: str) -> None:
    catalog = tmp_path / "catalog"
    _skill(catalog, "simple")
    unsafe = catalog / "unsafe"
    unsafe.mkdir()
    if kind == "symlink":
        external = tmp_path / "external.md"
        external.write_text("Missing frontmatter.\n", encoding="utf-8")
        (unsafe / "SKILL.md").symlink_to(external)
    else:
        # A manifest name must identify a regular file, even when it is a directory.
        (unsafe / "SKILL.md").mkdir()

    result = _validate(catalog, tmp_path / "reports")

    assert result.exit_code == 1, result.output
    assert "manifest must be a regular, unlinked file" in result.output
