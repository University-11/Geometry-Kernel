import tempfile
import unittest
from pathlib import Path

from geometry_kernel.datasets.movielens1m import iter_user_sequences, load_movielens_1m


class MovieLens1MTest(unittest.TestCase):
    def test_loads_raw_files_and_builds_chronological_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ratings.dat").write_text(
                "1::10::5::100\n"
                "1::20::3::90\n"
                "2::30::4::110\n",
                encoding="latin-1",
            )
            (root / "movies.dat").write_text(
                "10::Toy Story (1995)::Animation|Children's|Comedy\n"
                "20::Jumanji (1995)::Adventure|Children's|Fantasy\n"
                "30::Heat (1995)::Action|Crime|Thriller\n",
                encoding="latin-1",
            )
            (root / "users.dat").write_text(
                "1::F::1::10::48067\n"
                "2::M::25::20::55117\n",
                encoding="latin-1",
            )

            dataset = load_movielens_1m(root)

        self.assertEqual(len(dataset.ratings), 3)
        self.assertEqual(dataset.movies[10].genres, ("Animation", "Children's", "Comedy"))
        self.assertEqual(dataset.users[2].age, 25)
        self.assertEqual(list(dataset.user_sequences()), [(1, [20, 10]), (2, [30])])
        self.assertEqual(list(dataset.user_sequences(min_rating=4)), [(1, [10]), (2, [30])])

    def test_iter_user_sequences_is_deterministic_for_equal_timestamps(self) -> None:
        ratings = load_fake_ratings()
        self.assertEqual(list(iter_user_sequences(ratings)), [(1, [10, 20])])


def load_fake_ratings():
    from geometry_kernel.datasets.movielens1m import Rating

    return [
        Rating(user_id=1, movie_id=20, rating=5, timestamp=100),
        Rating(user_id=1, movie_id=10, rating=5, timestamp=100),
    ]


if __name__ == "__main__":
    unittest.main()
