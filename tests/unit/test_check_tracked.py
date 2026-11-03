from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]


def _guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_tracked", REPO / "scripts" / "check_tracked.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_rules(tmp_path: Path) -> None:
    guard = _guard()
    (tmp_path / "results" / "tables").mkdir(parents=True)
    (tmp_path / "results" / "tables" / "ok.csv").write_text("year,mean\n2001,0.1\n")
    (tmp_path / "results" / "tables" / "bad.csv").write_text("date,ticker,ret\n")
    (tmp_path / "big.bin").write_bytes(b"0" * 500_001)
    (tmp_path / "uv.lock").write_bytes(b"0" * 500_001)
    names = [
        "data/raw/x.json",
        "src/x.parquet",
        "tests/fixtures/synth.parquet",
        "notes.csv",
        "tests/fixtures/targets.csv",
        "results/trial_log.csv",
        "results/tables/ok.csv",
        "results/tables/bad.csv",
        "big.bin",
        "uv.lock",
        "src\\statarb\\cli.py",
    ]
    found = guard.problems(names, root=tmp_path)
    flagged = {message.split(":")[0] for message in found}
    assert flagged == {
        "data/raw/x.json",
        "src/x.parquet",
        "notes.csv",
        "results/tables/bad.csv",
        "big.bin",
    }
    assert guard.main([]) == 0


def test_repository_passes_guard() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    assert _guard().problems(tracked, root=REPO) == []
