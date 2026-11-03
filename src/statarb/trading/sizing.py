"""Position sizing (FR-15).

Committed capital is 1; each of the ``top_k`` pair slots holds ``1 / top_k`` of it, idle
when fewer pairs are selected. A pair's gross notional is ``gross_per_pair`` slots times
its scale. Distance pairs are dollar-neutral; cointegration pairs hold ``beta`` dollars of
``x`` per dollar of ``y``, split so the two legs' gross is the pair's gross.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

Scheme = Literal["equal_gross", "inverse_half_life", "vol_target"]


def pair_scales(
    half_lives: Sequence[float | None], spread_sigmas: Sequence[float], scheme: Scheme
) -> list[float]:
    """Relative size of each selected pair, averaging 1 across the pairs.

    ``inverse_half_life`` sizes by ``1 / half-life``, so faster-reverting spreads get more
    capital; ``vol_target`` by ``1 / spread sigma``, equal risk per pair. A pair without the
    needed estimate gets the average size.
    """
    count = len(half_lives)
    if count == 0:
        return []
    if scheme == "equal_gross":
        return [1.0] * count
    raw: list[float | None]
    if scheme == "inverse_half_life":
        raw = [
            1.0 / h if h is not None and h > 0 and math.isfinite(h) else None for h in half_lives
        ]
    else:
        raw = [1.0 / s if s > 0 and math.isfinite(s) else None for s in spread_sigmas]
    known = [value for value in raw if value is not None]
    if not known:
        return [1.0] * count
    mean = sum(known) / len(known)
    filled = [value / mean if value is not None else 1.0 for value in raw]
    total = sum(filled)
    return [value * count / total for value in filled]


def leg_weights(
    direction: int, beta: float, gross: float, dollar_neutral: bool
) -> tuple[float, float]:
    """Signed notionals of ``(y, x)`` for a spread position ``direction`` (+1 long y)."""
    if direction == 0:
        return 0.0, 0.0
    if dollar_neutral:
        return direction * gross / 2.0, -direction * gross / 2.0
    share = 1.0 + abs(beta)
    return direction * gross / share, -direction * gross * beta / share
