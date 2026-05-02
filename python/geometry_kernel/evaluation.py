"""Chronological recommendation evaluation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ChronologicalSplit:
    user_id: int
    train: list[int]
    test: list[int]


def chronological_split(
    sequences: Iterable[tuple[int, Sequence[int]]],
    train_fraction: float = 0.8,
    min_train: int = 1,
    min_test: int = 1,
) -> list[ChronologicalSplit]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if min_train < 0 or min_test < 0:
        raise ValueError("min_train and min_test must be non-negative")

    out: list[ChronologicalSplit] = []
    for user_id, seq in sequences:
        items = list(seq)
        if len(items) < min_train + min_test:
            continue
        split_at = int(len(items) * train_fraction)
        split_at = max(min_train, min(split_at, len(items) - min_test))
        out.append(ChronologicalSplit(user_id=user_id, train=items[:split_at], test=items[split_at:]))
    return out


def hit_rate_at_k(recommended: Sequence[int], relevant: int | set[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = {relevant} if isinstance(relevant, int) else set(relevant)
    return 1.0 if any(item in relevant_set for item in recommended[:k]) else 0.0


def ndcg_at_k(recommended: Sequence[int], relevant: int | set[int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = {relevant} if isinstance(relevant, int) else set(relevant)
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            return 1.0 / math.log2(rank + 1)
    return 0.0

