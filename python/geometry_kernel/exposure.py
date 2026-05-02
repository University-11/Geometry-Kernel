"""Exposure-aware sampling helpers for MovieLens-style logs."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

from .datasets.movielens1m import Rating


@dataclass(frozen=True)
class ExposureEvent:
    user_id: int
    positive_movie_id: int
    exposed_negative_ids: tuple[int, ...]
    unknown_negative_ids: tuple[int, ...]
    timestamp: int


def iter_exposure_events(
    ratings: Iterable[Rating],
    all_movie_ids: Sequence[int],
    min_positive_rating: int = 4,
    max_exposed_negatives: int = 4,
    max_unknown_negatives: int = 8,
    seed: int = 0,
) -> Iterator[ExposureEvent]:
    """Yield ternary feedback events: chosen, exposed-but-not-chosen, unknown."""

    if max_exposed_negatives < 0 or max_unknown_negatives < 0:
        raise ValueError("negative sample counts must be non-negative")

    rng = random.Random(seed)
    all_items = tuple(dict.fromkeys(all_movie_ids))
    by_user: dict[int, list[Rating]] = defaultdict(list)
    for rating in ratings:
        by_user[rating.user_id].append(rating)

    for user_id in sorted(by_user):
        history = sorted(by_user[user_id], key=lambda item: (item.timestamp, item.movie_id))
        seen = {rating.movie_id for rating in history}
        unknown_pool = [movie_id for movie_id in all_items if movie_id not in seen]
        exposed_pool: list[int] = []

        for rating in history:
            if rating.rating >= min_positive_rating:
                exposed = tuple(exposed_pool[-max_exposed_negatives:]) if max_exposed_negatives else ()
                unknown = _sample_without_replacement(rng, unknown_pool, max_unknown_negatives)
                yield ExposureEvent(
                    user_id=user_id,
                    positive_movie_id=rating.movie_id,
                    exposed_negative_ids=exposed,
                    unknown_negative_ids=unknown,
                    timestamp=rating.timestamp,
                )
            else:
                exposed_pool.append(rating.movie_id)


def _sample_without_replacement(rng: random.Random, values: Sequence[int], count: int) -> tuple[int, ...]:
    if count <= 0 or not values:
        return ()
    if count >= len(values):
        sample = list(values)
        rng.shuffle(sample)
        return tuple(sample)
    return tuple(rng.sample(list(values), count))

