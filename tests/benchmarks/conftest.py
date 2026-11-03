"""Benchmark settings: keep the host name out of reports/benchmarks.json."""

from __future__ import annotations

from typing import Any


def pytest_benchmark_update_machine_info(config: Any, machine_info: dict[str, Any]) -> None:
    """Drop the host name; CPU, OS and Python stay (docs/hardware.md has the rest)."""
    machine_info.pop("node", None)
