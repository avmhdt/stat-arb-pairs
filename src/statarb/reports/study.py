"""Every table and figure the memo cites, rebuilt from the study's runs (FR-29).

``build_all`` resolves the names in ``configs/study.yaml`` to complete runs and writes the
tables under ``results/tables`` and the figures under ``reports/figures``. It also writes
``study_runs.csv`` (name to run id) so every number can be traced to its run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from statarb.data.schemas import read_build_info
from statarb.reports import economics, figures
from statarb.reports.tables import write_fdr_outputs
from statarb.study import Study, resolve_all

TABLES = Path("results") / "tables"
FIGURES = Path("reports") / "figures"


@dataclass(frozen=True)
class Paths:
    """Where the report reads and writes."""

    results: Path
    processed: Path
    raw: Path
    tables: Path = TABLES
    figures: Path = FIGURES
    doc: Path = Path("docs") / "multiple_testing.md"

    @property
    def runs(self) -> Path:
        return self.results / "runs"

    @property
    def log(self) -> Path:
        return self.results / "trial_log.csv"


def paths_for(results: Path, data_root: Path, profile: str, doc: Path | None = None) -> Paths:
    """Headline outputs go to ``results/tables`` and ``reports/figures``; prototype outputs
    to their ``prototype/`` subfolders, so a prototype run never overwrites the memo's."""
    sub = Path() if profile == "headline" else Path("prototype")
    default_doc = Path("docs") / "multiple_testing.md"
    return Paths(
        results,
        data_root / "processed",
        data_root / "raw",
        tables=TABLES / sub,
        figures=FIGURES / sub,
        doc=doc or (default_doc if profile == "headline" else TABLES / sub / "fdr_summary.md"),
    )


def replication_tables(
    paths: Paths,
    ids: list[str],
    processed: Path | None,
    out: Path | None = None,
    figure: Path | None = None,
) -> dict[str, object]:
    """``ggr_replication.csv``/``.json``, ``ggr_bounce.csv`` and the replication figure."""
    from statarb.config import RunConfig
    from statarb.evaluation import replication
    from statarb.trading.backtest import load_for

    loaded = [replication.read_run(paths.runs, rid) for rid in ids]
    table = replication.replication_table(loaded, pl.read_csv(replication.TARGETS))
    out = out or paths.tables / "ggr_replication.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(out, float_precision=6)
    replication.figure(table, figure or paths.figures / "ggr_replication")
    result: dict[str, object] = replication.verdict(table)
    out.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    if processed is not None and {run.wait for run in loaded} == {"no_wait", "wait_one_day"}:
        import yaml

        params = yaml.safe_load((paths.runs / ids[0] / "config.yaml").read_text(encoding="utf-8"))
        for key in ("data_profile", "manifest_hash"):
            params.pop(key, None)
        data = load_for(RunConfig.model_validate(params), processed)
        bounce = replication.bounce_table(paths.runs, loaded, table, data)
        bounce.write_csv(out.with_name("ggr_bounce.csv"), float_precision=6)
    return result


def al_tables(paths: Paths, ids: list[str]) -> None:
    """``al_vs_pairs*.csv``, ``al_vs_paper.csv`` and the A-L figure (AC-H-03)."""
    from statarb.evaluation import comparison

    loaded = [comparison.read_run(paths.runs, rid) for rid in ids]
    paths.tables.mkdir(parents=True, exist_ok=True)
    comparison.summary(loaded).write_csv(paths.tables / "al_vs_pairs.csv", float_precision=6)
    yearly = comparison.by_year(loaded)
    yearly.write_csv(paths.tables / "al_vs_pairs_by_year.csv", float_precision=6)
    comparison.differences(loaded).write_csv(
        paths.tables / "al_vs_pairs_sharpe_diff.csv", float_precision=6
    )
    comparison.against_paper(loaded, pl.read_csv(comparison.TARGETS)).write_csv(
        paths.tables / "al_vs_paper.csv", float_precision=6
    )
    comparison.figure(yearly, paths.figures / "al_vs_pairs")


def protocol_figure(path: Path) -> None:
    """The rolling formation/trading protocol and the per-window pipeline."""
    plt = figures._plt()
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [1.1, 1.0]}
    )
    for row in range(4):
        start = row
        top.barh(row, 12, left=start, color="#4c72b0", height=0.6)
        top.barh(row, 6, left=start + 12, color="#dd8452", height=0.6)
        top.text(start + 6, row, "formation (12 months)", ha="center", va="center", color="white")
        top.text(start + 15, row, "trading (6)", ha="center", va="center", color="white")
        top.plot([start + 12, start + 12], [row - 0.4, row + 0.4], color="black", linewidth=1)
        top.text(start + 12.1, row - 0.32, "T", fontsize=8)
    top.set_yticks(range(4), [f"window {i + 1}" for i in range(4)])
    top.invert_yaxis()
    top.set_xlabel("months; a new window starts every month, six trade at once")
    top.set_xlim(-0.5, 22)
    top.set_title("Rolling windows: everything up to T selects and fits; after T only trades")
    steps = [
        "universe\nat T - 1",
        "candidates\n(SIC2 groups)",
        "Engle-Granger\nboth orderings",
        "BH at q = 0.05",
        "half-life band\n+ Johansen",
        "freeze params,\nKalman state",
        "trade with\nlag 1, costs",
    ]
    for i, label in enumerate(steps):
        bottom.add_patch(
            plt.Rectangle((i * 1.45, 0.2), 1.25, 0.6, facecolor="#eeeeee", edgecolor="#333333")
        )
        bottom.text(i * 1.45 + 0.625, 0.5, label, ha="center", va="center", fontsize=8)
        if i < len(steps) - 1:
            bottom.annotate(
                "",
                xy=(i * 1.45 + 1.45, 0.5),
                xytext=(i * 1.45 + 1.25, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    bottom.set_xlim(-0.1, len(steps) * 1.45)
    bottom.set_ylim(0, 1)
    bottom.axis("off")
    bottom.set_title("Per window (cointegration method)", fontsize=10)
    fig.tight_layout()
    figures.save(fig, path)


def example_pair_figure(path: Path, delta: float = 1e-4, seed: int = 7) -> None:
    """Kalman hedge ratio, its standardised innovation z and any trades on a synthetic pair.

    Synthetic by design: the licensed data may only be published as aggregates. At the
    spec's delta the innovation z is underdispersed and rarely reaches the entry level,
    which is the point the memo makes with this figure.
    """
    from statarb.models.kalman import formation_start, kalman_filter
    from statarb.trading.rules import state_machine

    rng = np.random.default_rng(seed)
    n_form, n_trade = 252, 126
    n = n_form + n_trade
    # Log prices start at 0 so a drifting beta does not masquerade as a level shift; spread
    # shocks are fat-tailed (Student t, 4 degrees of freedom) as daily returns are.
    x = np.cumsum(rng.normal(0.0, 0.015, n))
    beta = 0.9 + 0.15 * np.sin(np.linspace(0, 3, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.9 * spread[t - 1] + 0.004 * rng.standard_t(4)
    y = 0.02 + beta * x + spread
    theta0, p0, obs_var = formation_start(y[:n_form], x[:n_form])
    result, _ = kalman_filter(y, x, delta, obs_var, theta0, p0)
    z = result.z[n_form:]
    position, _ = state_machine(z, 2.0, 0.0, 4.0, "cross_zero", None)
    plt = figures._plt()
    fig, (ax_beta, ax_z) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    days = np.arange(n) - n_form
    ax_beta.plot(days, beta, color="#999999", label="true beta")
    ax_beta.plot(days, result.beta, color="#4c72b0", label=f"Kalman beta (delta = {delta:g})")
    ax_beta.axvline(0, color="black", linewidth=0.8)
    ax_beta.set_ylabel("hedge ratio")
    ax_beta.legend(fontsize=7)
    ax_beta.set_title(
        "Synthetic pair at the spec's delta: the Kalman hedge tracks beta, but z stays inside +/-2"
    )
    ax_z.plot(days[n_form:], z, color="#333333", linewidth=1)
    for level in (-2.0, 2.0):
        ax_z.axhline(level, color="#dd8452", linewidth=0.8, linestyle="--")
    ax_z.axhline(0, color="black", linewidth=0.6)
    changes = np.nonzero(np.diff(np.r_[0, position]))[0]
    for t in changes:
        opened = position[t] != 0
        ax_z.scatter(
            days[n_form + t],
            z[t],
            marker="^" if opened else "o",
            color="#55a868" if opened else "#c44e52",
            zorder=3,
        )
    ax_z.set_ylabel("z (trading window)")
    ax_z.set_xlabel("sessions from T (formation before 0)")
    fig.tight_layout()
    figures.save(fig, path)


def build_all(study: Study, paths: Paths, profile: str = "headline") -> dict[str, str]:
    """Resolve the study and rebuild every table and figure; returns name to run id."""
    info = read_build_info(paths.processed)
    manifest = str(info.get("manifest_hash", ""))
    ids = resolve_all(study, paths.log, paths.runs, profile, manifest)
    t = study.tables
    paths.tables.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"name": list(ids), "run_id": list(ids.values()), "data_profile": profile}
    ).write_csv(paths.tables / "study_runs.csv")

    def labelled(mapping: dict[str, str]) -> list[str]:
        return [f"{label}={ids[name]}" for label, name in mapping.items()]

    def plain(names: list[str]) -> list[str]:
        return [ids[name] for name in names]

    if "fdr" in t:
        fdr_name = study.runs[t["fdr"]].resolve().run.name
        write_fdr_outputs(
            {fdr_name: paths.runs / ids[t["fdr"]] / "pairs.parquet"},
            paths.tables,
            paths.doc,
            profile,
        )
    if "replication" in t:
        replication_tables(paths, plain(t["replication"]), paths.processed)
    if "al_comparison" in t:
        al_tables(paths, plain(t["al_comparison"]))
    if "decay" in t:
        economics.decay(paths.runs, labelled(t["decay"]), paths.tables, paths.figures)
    if "persistence" in t:
        economics.persistence_tables(
            paths.runs, paths.processed, labelled(t["persistence"]), paths.tables, paths.figures
        )
    if "economics" in t:
        specs = labelled(t["economics"])
        economics.break_even_tables(paths.runs, specs, paths.tables, paths.figures)
        economics.capacity_tables(paths.runs, specs, paths.tables, paths.figures)
        factors = paths.raw / "french" / "factors.parquet"
        if factors.exists():
            economics.crowding_tables(paths.runs, factors, specs, paths.tables, paths.figures)
    if "policies" in t:
        economics.policy_table(paths.runs, plain(t["policies"]), paths.tables)
    if "sizing" in t:
        economics.sizing_table(paths.runs, plain(t["sizing"]), paths.tables, paths.figures)
    if "deflated_sharpe" in t:
        economics.dsr_table(
            paths.log,
            {family: ids[name] for family, name in t["deflated_sharpe"].items()},
            paths.tables,
        )
    if "survivorship" in t:
        economics.survivorship_table(
            paths.runs, [(ids[a], ids[b]) for a, b in t["survivorship"]], paths.tables
        )
    snapshots = sorted((paths.raw / "ibkr").glob("snapshot=*"))
    if snapshots and profile == "headline":
        from statarb.data.build import borrow_rates

        economics.borrow_table(
            snapshots[-1],
            paths.processed,
            borrow_rates(Path("configs") / "borrow_schedule.yaml"),
            paths.tables,
        )
    if "kalman" in t:
        economics.kalman_table(paths.runs, plain(t["kalman"]), paths.tables)
    if profile == "headline":
        protocol_figure(paths.figures / "protocol")
        example_pair_figure(paths.figures / "example_pair")
    return ids
