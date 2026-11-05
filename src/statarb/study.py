"""The study manifest: every run the memo cites, and how to find it again (FR-29).

``configs/study.yaml`` names each run by its config and overrides. A name resolves to the
newest complete run in the trial log whose resolved config, data profile and data manifest
match; "complete" means its folder holds ``metrics.json``, which a run writes last.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from statarb.config import RunConfig, load_config

STUDY = Path("configs") / "study.yaml"
PROVENANCE = ("data_profile", "manifest_hash")


class StudyError(RuntimeError):
    """A study name is unknown or has no matching complete run."""


@dataclass(frozen=True)
class RunSpec:
    """One named run: a config file and its ``--set`` overrides."""

    name: str
    config: Path
    overrides: tuple[str, ...]

    def resolve(self) -> RunConfig:
        return load_config(self.config, list(self.overrides))


@dataclass(frozen=True)
class Study:
    """The parsed manifest."""

    runs: dict[str, RunSpec]
    tables: dict[str, Any]

    def names(self, key: str) -> list[str]:
        """Every run name a table entry refers to, in order."""
        found: list[str] = []

        def walk(value: object) -> None:
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(self.tables[key])
        return found


def load_study(path: Path = STUDY) -> Study:
    """Parse ``path``; every name a table uses must be defined under ``runs``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    runs = {
        name: RunSpec(name, Path(spec["config"]), tuple(str(item) for item in spec.get("set", [])))
        for name, spec in (raw.get("runs") or {}).items()
    }
    study = Study(runs, dict(raw.get("tables") or {}))
    unknown = sorted({n for key in study.tables for n in study.names(key)} - set(runs))
    if unknown:
        raise StudyError(f"tables refer to undefined runs: {unknown}")
    return study


def normalized(params: dict[str, Any]) -> dict[str, Any]:
    """A run's config as the current model dumps it, provenance removed.

    Validating through ``RunConfig`` fills keys added after the run was made with their
    defaults, so older runs still match a config that leaves those keys alone.
    """
    clean = {key: value for key, value in params.items() if key not in PROVENANCE}
    dumped: dict[str, Any] = json.loads(RunConfig.model_validate(clean).model_dump_json())
    return dumped


def find_run(
    log_path: Path, runs_dir: Path, cfg: RunConfig, profile: str, manifest_hash: str
) -> str | None:
    """The newest complete trial whose config, profile and manifest match, if any."""
    if not log_path.exists():
        return None
    target = json.loads(cfg.model_dump_json())
    with log_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in reversed(rows):
        if row.get("n_obs") in (None, "", "0"):
            continue
        params = json.loads(row["params_json"])
        if params.get("data_profile") != profile or params.get("manifest_hash") != manifest_hash:
            continue
        if not (runs_dir / row["run_id"] / "metrics.json").exists():
            continue
        if normalized(params) == target:
            return str(row["run_id"])
    return None


def resolve_all(
    study: Study, log_path: Path, runs_dir: Path, profile: str, manifest_hash: str
) -> dict[str, str]:
    """Run id of every named run; raises ``StudyError`` naming the ones with no match."""
    found: dict[str, str] = {}
    missing = []
    for name, spec in study.runs.items():
        run_id = find_run(log_path, runs_dir, spec.resolve(), profile, manifest_hash)
        if run_id is None:
            missing.append(name)
        else:
            found[name] = run_id
    if missing:
        raise StudyError(
            f"no complete {profile} run for {missing}; run them with `statarb reproduce`"
        )
    return found
