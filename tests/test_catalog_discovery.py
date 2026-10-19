# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Catalog runs must account for every live supported skill manifest."""

import errno
import json
import os
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


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("filename", ["SKILL.md", "skill.md"])
def test_catalog_stops_at_skill_boundary(tmp_path: Path, workers: int, filename: str) -> None:
    catalog = tmp_path / "catalog"
    skill = _skill(catalog, "team/simple", filename=filename)
    _skill(skill, "assets/templates/new-skill", valid=False)
    reports = tmp_path / "reports"

    assert find_skills_in_directory(catalog) == [skill]
    result = _validate(catalog, reports, workers)

    expected_passed = filename == "SKILL.md"
    assert result.exit_code == (0 if expected_passed else 1), result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["skills"][0]["name"] == "team/simple"
    assert summary["overall_passed"] is expected_passed


def test_discovery_reports_directory_links_without_following_them(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    live = _skill(catalog, "simple")
    external = _skill(tmp_path, "external", valid=False)
    linked = catalog / "linked"
    linked.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match=r"[Ll]ink"):
        find_skills_in_directory(catalog)

    problems: dict[Path, str] = {}
    assert find_skills_in_directory(catalog, problems=problems) == sorted([live, linked])
    assert set(problems) == {linked}
    assert "link" in problems[linked].lower()


def test_discovery_records_unreadable_linked_manifest_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = tmp_path / "catalog"
    live = _skill(catalog, "simple")
    external = _skill(tmp_path, "external")
    linked = catalog / "linked"
    linked.symlink_to(external, target_is_directory=True)
    real_exists = Path.exists

    def deny_linked_manifest(path: Path, *args, **kwargs):
        if path == linked / "SKILL.md":
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_exists(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", deny_linked_manifest)
    with pytest.raises(ValueError, match="Permission denied"):
        find_skills_in_directory(catalog)

    problems: dict[Path, str] = {}
    assert find_skills_in_directory(catalog, problems=problems) == sorted([live, linked])
    assert set(problems) == {linked}
    assert "Cannot inspect linked skill directory" in problems[linked]
    assert "Permission denied" in problems[linked]


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("include_live", [False, True])
def test_catalog_records_linked_skills_as_failed_entries(tmp_path: Path, workers: int, include_live: bool) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    if include_live:
        _skill(catalog, "simple")
    external = _skill(tmp_path, "external")
    (catalog / "linked").symlink_to(external, target_is_directory=True)
    reports = tmp_path / "reports"

    result = _validate(catalog, reports, workers)

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1 + include_live
    assert summary["passed"] == int(include_live)
    assert summary["failed"] == 1
    assert summary["overall_passed"] is False
    entries = {entry["name"]: entry for entry in summary["skills"]}
    linked = entries["linked"]
    assert linked["passed"] is False
    assert "link" in linked["reason"].lower()
    assert "linked" in linked["reason"]
    assert "json_report" not in linked
    if include_live:
        assert entries["simple"]["passed"] is True
        assert (reports / "simple" / entries["simple"]["json_report"]).is_file()


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_catalog_accepts_linked_manifest_files(tmp_path: Path, kind: str) -> None:
    catalog = tmp_path / "catalog"
    skill = _skill(catalog, "simple")
    manifest = skill / "SKILL.md"
    external = tmp_path / "external.md"
    manifest.rename(external)
    if kind == "symlink":
        manifest.symlink_to(external)
    else:
        manifest.hardlink_to(external)
    reports = tmp_path / "reports"

    assert find_skills_in_directory(manifest) == [skill]
    assert find_skills_in_directory(catalog) == [skill]
    result = _validate(catalog, reports)

    assert result.exit_code == 0, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1
    assert summary["overall_passed"] is True


def test_catalog_records_nonregular_manifest_failure_and_completes_other_skills(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    _skill(catalog, "simple")
    (catalog / "broken" / "SKILL.md").mkdir(parents=True)
    reports = tmp_path / "reports"

    result = _validate(catalog, reports)

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in summary["skills"]}
    assert set(entries) == {"simple", "broken"}
    assert entries["simple"]["passed"] is True
    assert entries["broken"]["passed"] is False
    assert summary["overall_passed"] is False


@pytest.mark.parametrize("workers", [1, 2])
def test_catalog_records_unreadable_directory_and_completes_other_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    catalog = tmp_path / "catalog"
    live = _skill(catalog, "simple")
    unreadable = _skill(catalog, "blocked")
    real_scandir = os.scandir

    def deny_blocked_directory(path):
        if str(path) == str(unreadable):
            raise PermissionError(errno.EACCES, "Permission denied", str(unreadable))
        return real_scandir(path)

    monkeypatch.setattr("skillevaluator.utils.helpers.os.scandir", deny_blocked_directory)
    with pytest.raises(ValueError, match="Permission denied"):
        find_skills_in_directory(catalog)
    problems: dict[Path, str] = {}
    assert find_skills_in_directory(catalog, problems=problems) == sorted([live, unreadable])
    assert set(problems) == {unreadable}
    reports = tmp_path / "reports"

    result = _validate(catalog, reports, workers)

    assert result.exit_code == 1, result.output
    summary = json.loads((reports / "catalog-summary.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in summary["skills"]}
    assert set(entries) == {"simple", "blocked"}
    assert entries["simple"]["passed"] is True
    assert (reports / "simple" / entries["simple"]["json_report"]).is_file()
    assert entries["blocked"]["passed"] is False
    assert "blocked" in entries["blocked"]["reason"]
    assert "Permission denied" in entries["blocked"]["reason"]
    assert "json_report" not in entries["blocked"]
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["overall_passed"] is False
