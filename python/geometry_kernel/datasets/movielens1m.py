"""Parser for the raw MovieLens-1M dataset.

MovieLens-1M raw files use ``::`` as a separator:

* ``ratings.dat``: ``UserID::MovieID::Rating::Timestamp``
* ``movies.dat``: ``MovieID::Title::Genres``
* ``users.dat``: ``UserID::Gender::Age::Occupation::Zip-code``

The functions below are intentionally small and dependency-free. They stream
records when possible and leave modeling decisions to downstream code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, Iterator


DEFAULT_ENCODING = "latin-1"


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    rating: int
    timestamp: int


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    genres: tuple[str, ...]


@dataclass(frozen=True)
class User:
    user_id: int
    gender: str
    age: int
    occupation: int
    zip_code: str


@dataclass(frozen=True)
class MovieLens1M:
    ratings: list[Rating]
    movies: dict[int, Movie]
    users: dict[int, User]

    def user_sequences(self, min_rating: int | None = None) -> Iterator[tuple[int, list[int]]]:
        return iter_user_sequences(self.ratings, min_rating=min_rating)


def iter_ratings(path: str | Path, encoding: str = DEFAULT_ENCODING) -> Iterator[Rating]:
    for line_no, parts in _iter_parts(path, expected_parts=4, encoding=encoding):
        try:
            yield Rating(
                user_id=int(parts[0]),
                movie_id=int(parts[1]),
                rating=int(parts[2]),
                timestamp=int(parts[3]),
            )
        except ValueError as exc:
            raise ValueError(f"Invalid rating at line {line_no} in {path!s}: {parts!r}") from exc


def iter_movies(path: str | Path, encoding: str = DEFAULT_ENCODING) -> Iterator[Movie]:
    for line_no, parts in _iter_parts(path, expected_parts=3, encoding=encoding):
        try:
            movie_id = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"Invalid movie id at line {line_no} in {path!s}: {parts[0]!r}") from exc

        genres = tuple(genre for genre in parts[2].split("|") if genre)
        yield Movie(movie_id=movie_id, title=parts[1], genres=genres)


def iter_users(path: str | Path, encoding: str = DEFAULT_ENCODING) -> Iterator[User]:
    for line_no, parts in _iter_parts(path, expected_parts=5, encoding=encoding):
        try:
            yield User(
                user_id=int(parts[0]),
                gender=parts[1],
                age=int(parts[2]),
                occupation=int(parts[3]),
                zip_code=parts[4],
            )
        except ValueError as exc:
            raise ValueError(f"Invalid user at line {line_no} in {path!s}: {parts!r}") from exc


def load_movielens_1m(root: str | Path, encoding: str = DEFAULT_ENCODING) -> MovieLens1M:
    root_path = Path(root)
    ratings = list(iter_ratings(root_path / "ratings.dat", encoding=encoding))
    movies = {movie.movie_id: movie for movie in iter_movies(root_path / "movies.dat", encoding=encoding)}
    users = {user.user_id: user for user in iter_users(root_path / "users.dat", encoding=encoding)}
    return MovieLens1M(ratings=ratings, movies=movies, users=users)


def iter_user_sequences(
    ratings: Iterable[Rating],
    min_rating: int | None = None,
) -> Iterator[tuple[int, list[int]]]:
    """Yield ``(user_id, movie_id_sequence)`` sorted by interaction time.

    ``min_rating`` can be used to keep only positive feedback, for example
    ``min_rating=4`` for ratings 4 and 5.
    """

    grouped: DefaultDict[int, list[Rating]] = defaultdict(list)
    for rating in ratings:
        if min_rating is not None and rating.rating < min_rating:
            continue
        grouped[rating.user_id].append(rating)

    for user_id in sorted(grouped):
        ordered = sorted(grouped[user_id], key=lambda item: (item.timestamp, item.movie_id))
        yield user_id, [item.movie_id for item in ordered]


def _iter_parts(path: str | Path, expected_parts: int, encoding: str) -> Iterator[tuple[int, list[str]]]:
    file_path = Path(path)
    with file_path.open("r", encoding=encoding) as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != expected_parts:
                raise ValueError(
                    f"Expected {expected_parts} fields at line {line_no} in {file_path!s}, "
                    f"got {len(parts)}: {line!r}"
                )
            yield line_no, parts

