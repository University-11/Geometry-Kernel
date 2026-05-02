import unittest

from geometry_kernel.evaluation import chronological_split, hit_rate_at_k, ndcg_at_k


class EvaluationTest(unittest.TestCase):
    def test_chronological_split_preserves_order(self) -> None:
        splits = chronological_split([(1, [10, 20, 30, 40, 50])], train_fraction=0.6)

        self.assertEqual(splits[0].train, [10, 20, 30])
        self.assertEqual(splits[0].test, [40, 50])

    def test_ranking_metrics(self) -> None:
        recommended = [10, 20, 30]

        self.assertEqual(hit_rate_at_k(recommended, 20, 2), 1.0)
        self.assertEqual(hit_rate_at_k(recommended, 30, 2), 0.0)
        self.assertAlmostEqual(ndcg_at_k(recommended, 20, 3), 1.0 / 1.584962500721156)


if __name__ == "__main__":
    unittest.main()

