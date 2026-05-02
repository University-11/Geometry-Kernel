"""Basic Poincare ball operations for the GeoMemory Kernel.

The document in ``base/`` describes GeoMemory items as tangent-space codes
mapped with ``exp0`` and compared with Poincare distance. Here ``c`` is the
positive curvature scale, so the manifold curvature is ``-c``. Setting ``c=0``
uses the Euclidean limit.
"""

from __future__ import annotations

import math
from typing import Sequence


Vector = Sequence[float]
EPS = 1e-12


def mobius_add(x: Vector, y: Vector, c: float = 1.0) -> list[float]:
    """Return the Mobius sum ``x (+)_c y`` in the Poincare ball."""

    _validate_curvature(c)
    _validate_same_dim(x, y)
    if c == 0.0:
        return [float(a) + float(b) for a, b in zip(x, y)]

    x2 = _squared_norm(x)
    y2 = _squared_norm(y)
    xy = _dot(x, y)

    first = 1.0 + 2.0 * c * xy + c * y2
    second = 1.0 - c * x2
    denominator = 1.0 + 2.0 * c * xy + (c * c) * x2 * y2
    if abs(denominator) < EPS:
        denominator = EPS if denominator >= 0.0 else -EPS

    result = [
        (first * float(a) + second * float(b)) / denominator
        for a, b in zip(x, y)
    ]
    return project_to_ball(result, c=c)


def exp0(v: Vector, c: float = 1.0) -> list[float]:
    """Map a tangent vector at the origin to the Poincare ball."""

    _validate_curvature(c)
    if c == 0.0:
        return [float(value) for value in v]

    norm_v = _norm(v)
    if norm_v < EPS:
        return [0.0 for _ in v]

    sqrt_c = math.sqrt(c)
    scale = math.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v)
    return project_to_ball([scale * float(value) for value in v], c=c)


def poincare_distance(x: Vector, y: Vector, c: float = 1.0) -> float:
    """Return geodesic distance between two points in the Poincare ball."""

    _validate_curvature(c)
    _validate_same_dim(x, y)
    if c == 0.0:
        return _norm([float(a) - float(b) for a, b in zip(x, y)])

    diff = mobius_add([-float(value) for value in x], y, c=c)
    arg = math.sqrt(c) * _norm(diff)
    arg = min(max(arg, 0.0), 1.0 - EPS)
    return 2.0 * math.atanh(arg) / math.sqrt(c)


def project_to_ball(x: Vector, c: float = 1.0, eps: float = 1e-9) -> list[float]:
    """Project a vector inside the open Poincare ball for numerical safety."""

    _validate_curvature(c)
    values = [float(value) for value in x]
    if c == 0.0:
        return values

    norm_x = _norm(values)
    max_norm = (1.0 - eps) / math.sqrt(c)
    if norm_x <= max_norm:
        return values
    scale = max_norm / max(norm_x, EPS)
    return [scale * value for value in values]


def _validate_curvature(c: float) -> None:
    if c < 0.0:
        raise ValueError("c must be non-negative")


def _validate_same_dim(x: Vector, y: Vector) -> None:
    if len(x) != len(y):
        raise ValueError(f"Vector dimensions differ: {len(x)} != {len(y)}")


def _dot(x: Vector, y: Vector) -> float:
    return sum(float(a) * float(b) for a, b in zip(x, y))


def _squared_norm(x: Vector) -> float:
    return _dot(x, x)


def _norm(x: Vector) -> float:
    return math.sqrt(_squared_norm(x))

