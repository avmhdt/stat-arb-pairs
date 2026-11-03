"""Wide formation panels: one column per security over the formation sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

if TYPE_CHECKING:
    from statarb.windows import FormationView

F64 = npt.NDArray[np.float64]


def formation_panel(view: FormationView, ids: list[str], column: str = "tr_index") -> F64:
    """``(sessions, len(ids))`` values of ``column`` over the formation window.

    Missing days carry the last value forward (no trade, no change); a security with no
    value yet at the window's start keeps NaN there, and callers drop it.
    """
    sessions = pl.DataFrame({"date": view.sessions()})
    if not ids:
        return np.empty((sessions.height, 0))
    long = view.prices([column]).filter(pl.col("security_id").is_in(ids))
    wide = long.pivot(on="security_id", index="date", values=column)
    frame = sessions.join(wide, on="date", how="left").sort("date")
    missing = [name for name in ids if name not in frame.columns]
    if missing:
        frame = frame.with_columns(pl.lit(None, pl.Float64).alias(name) for name in missing)
    filled = frame.select(pl.col(ids).fill_nan(None).forward_fill())
    return filled.to_numpy().astype(np.float64)
