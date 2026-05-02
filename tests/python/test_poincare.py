import math
import unittest

from geometry_kernel.poincare import exp0, mobius_add, poincare_distance, project_to_ball


class PoincareTest(unittest.TestCase):
    def test_euclidean_limit(self) -> None:
        self.assertEqual(mobius_add([1.0, 2.0], [3.0, 4.0], c=0.0), [4.0, 6.0])
        self.assertEqual(exp0([1.0, -2.0], c=0.0), [1.0, -2.0])
        self.assertAlmostEqual(poincare_distance([0.0, 0.0], [3.0, 4.0], c=0.0), 5.0)

    def test_mobius_identity_and_distance_symmetry(self) -> None:
        x = [0.1, -0.2]
        y = [0.2, 0.05]
        self.assertEqual(mobius_add(x, [0.0, 0.0]), x)
        self.assertAlmostEqual(poincare_distance(x, y), poincare_distance(y, x), places=12)
        self.assertAlmostEqual(poincare_distance(x, x), 0.0, places=12)

    def test_exp0_maps_inside_ball(self) -> None:
        point = exp0([1.0, 0.0], c=1.0)
        self.assertAlmostEqual(point[0], math.tanh(1.0), places=12)
        self.assertAlmostEqual(point[1], 0.0, places=12)

    def test_distance_from_origin_matches_exp0_scale(self) -> None:
        tangent = [0.3, 0.4]
        point = exp0(tangent, c=1.0)
        self.assertAlmostEqual(poincare_distance([0.0, 0.0], point), 2.0 * 0.5, places=12)

    def test_projection_keeps_point_inside_open_ball(self) -> None:
        point = project_to_ball([10.0, 0.0], c=4.0)
        self.assertLess(math.sqrt(sum(value * value for value in point)), 0.5)


if __name__ == "__main__":
    unittest.main()
