"""Static figures for the memo: PNG at 300 dpi and SVG, matplotlib only (spec section 10)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save(fig: object, path: Path) -> None:
    """Write ``path.png`` and ``path.svg``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300)  # type: ignore[attr-defined]
    fig.savefig(path.with_suffix(".svg"))  # type: ignore[attr-defined]
    _plt().close(fig)


def lines(
    frame: pl.DataFrame,
    x: str,
    y: str,
    group: str,
    path: Path,
    title: str,
    ylabel: str,
    logx: bool = False,
    hline: float | None = 0.0,
) -> None:
    """One line per ``group`` value."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for (name,), part in frame.drop_nulls(y).group_by([group], maintain_order=True):
        part = part.sort(x)
        ax.plot(
            part[x].to_list(),
            part[y].to_list(),
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=str(name),
        )
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.6)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, path)


def bars(
    labels: Sequence[str],
    series: dict[str, Sequence[float]],
    path: Path,
    title: str,
    ylabel: str,
    errors: dict[str, tuple[Sequence[float], Sequence[float]]] | None = None,
) -> None:
    """Grouped bars, optionally with interval whiskers."""
    import numpy as np

    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.8 / max(len(series), 1)
    x = np.arange(len(labels))
    for i, (name, values) in enumerate(series.items()):
        offset = x - 0.4 + width * (i + 0.5)
        yerr = None
        if errors and name in errors:
            low, high = errors[name]
            yerr = np.vstack([np.array(values) - np.array(low), np.array(high) - np.array(values)])
        ax.bar(offset, values, width, label=name, yerr=yerr, capsize=2)
    ax.set_xticks(x, labels, fontsize=7, rotation=20)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, path)
