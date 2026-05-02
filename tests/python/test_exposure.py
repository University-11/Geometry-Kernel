import unittest

from geometry_kernel.datasets.movielens1m import Rating
from geometry_kernel.exposure import iter_exposure_events


class ExposureTest(unittest.TestCase):
    def test_exposure_events_separate_exposed_and_unknown_negatives(self) -> None:
        ratings = [
            Rating(1, 10, 2, 1),
            Rating(1, 20, 5, 2),
            Rating(1, 30, 1, 3),
            Rating(1, 40, 4, 4),
        ]

        events = list(
            iter_exposure_events(
                ratings,
                all_movie_ids=[10, 20, 30, 40, 50, 60],
                min_positive_rating=4,
                max_exposed_negatives=2,
                max_unknown_negatives=1,
                seed=7,
            )
        )

        self.assertEqual([event.positive_movie_id for event in events], [20, 40])
        self.assertEqual(events[0].exposed_negative_ids, (10,))
        self.assertEqual(events[1].exposed_negative_ids, (10, 30))
        self.assertEqual(len(events[0].unknown_negative_ids), 1)
        self.assertNotIn(events[0].unknown_negative_ids[0], {10, 20, 30, 40})


if __name__ == "__main__":
    unittest.main()
