"""Entry, exit and stop rules as a pure state machine (FR-14).

The position after the close of day ``t`` depends only on ``z[:t + 1]``:

- flat: enter when ``entry <= |z| < stop``, short the spread when ``z > 0``, long when
  ``z < 0``; never after a stop in the same window, and not when ``|z|`` is already past
  ``stop`` (entering would stop out at once);
- open: stop at ``|z| >= stop``; else exit when ``z`` reaches or crosses zero
  (``cross_zero``) or ``|z| <= exit`` (``band``); else time-stop after ``max_holding``
  sessions;
- never close and reopen on the same day, so a position never flips sign without a day
  at zero; a NaN ``z`` changes nothing;
- the last day is flat when ``close_at_end`` (window end);
- under a break policy, a flagged day closes the position and blocks entries (FR-26).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

F64 = npt.NDArray[np.float64]
I8 = npt.NDArray[np.int8]

NONE, ENTRY, EXIT, STOP, TIME_STOP, WINDOW_END, BREAK = 0, 1, 2, 3, 4, 5, 6
REASONS = {
    ENTRY: "entry",
    EXIT: "exit",
    STOP: "stop",
    TIME_STOP: "time_stop",
    WINDOW_END: "window_end",
    BREAK: "break",
}


@njit(cache=True, fastmath=False)
def _machine(
    z: F64,
    entry: float,
    exit_level: float,
    stop: float,
    band: bool,
    max_holding: int,
    close_at_end: bool,
    broken: npt.NDArray[np.bool_],
    break_close: bool,
    break_blacklist: bool,
) -> tuple[I8, I8]:
    n = z.size
    position = np.zeros(n, dtype=np.int8)
    reason = np.zeros(n, dtype=np.int8)
    state = 0
    held = 0
    stopped = False
    for t in range(n):
        value = z[t]
        if state != 0:
            held += 1
        if break_close and broken[t]:
            if state != 0:
                state = 0
                reason[t] = 6
            if break_blacklist:
                stopped = True
        elif not math.isnan(value):
            magnitude = abs(value)
            if state == 0:
                if not stopped and entry <= magnitude < stop:
                    state = -1 if value > 0 else 1
                    held = 0
                    reason[t] = 1
            elif magnitude >= stop:
                state = 0
                stopped = True
                reason[t] = 3
            elif (band and magnitude <= exit_level) or (not band and value * -state <= 0.0):
                state = 0
                reason[t] = 2
            elif max_holding > 0 and held >= max_holding:
                state = 0
                reason[t] = 4
        if close_at_end and t == n - 1 and state != 0:
            state = 0
            reason[t] = 5
        position[t] = state
    return position, reason


def state_machine(
    z: F64,
    entry: float,
    exit: float,
    stop: float | None,
    exit_mode: Literal["cross_zero", "band"],
    max_holding: int | None,
    close_at_end: bool = True,
    broken: npt.NDArray[np.bool_] | None = None,
    break_close: bool = False,
    break_blacklist: bool = False,
) -> tuple[I8, I8]:
    """Positions (-1, 0, +1) after each close and the reason code of each change.

    ``broken`` flags days on which the pair's cointegration test fails (FR-26). With
    ``break_close`` an open position closes on such a day and none opens; with
    ``break_blacklist`` the pair also stays closed for the rest of the window.
    """
    if exit_mode not in ("cross_zero", "band"):
        raise ValueError(f"unknown exit_mode {exit_mode!r}")
    stop_level = math.inf if stop is None else float(stop)
    if not 0.0 < entry < stop_level:
        raise ValueError("need 0 < entry < stop")
    return _machine(
        np.ascontiguousarray(z, dtype=np.float64),
        float(entry),
        float(exit),
        stop_level,
        exit_mode == "band",
        0 if max_holding is None else int(max_holding),
        close_at_end,
        np.zeros(len(z), dtype=np.bool_)
        if broken is None
        else np.ascontiguousarray(broken, dtype=np.bool_),
        break_close,
        break_blacklist,
    )


def position_state_machine(
    z: F64,
    entry: float,
    exit: float,
    stop: float,
    exit_mode: Literal["cross_zero", "band"],
    max_holding: int | None,
) -> I8:
    """The spec's interface: positions only, flat on the last day."""
    return state_machine(z, entry, exit, stop, exit_mode, max_holding)[0]


def max_holding_days(half_life: float | None, multiple: float | None) -> int | None:
    """Time stop in sessions: ``ceil(multiple * half_life)``, or None when either is missing."""
    if half_life is None or multiple is None or not math.isfinite(half_life):
        return None
    return max(1, math.ceil(multiple * half_life))
