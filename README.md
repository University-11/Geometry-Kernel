# Geometry Kernel

Geometry Kernel is the GeoMemory part of the three-kernel AI stack. It targets
recommendation and streaming personalization with explicit hyperbolic geometry,
exposure-aware feedback, and online user memory updates.

## What Belongs Here

- MovieLens-1M raw parser and chronological sequence helpers.
- Exposure-aware ternary event sampling: chosen, exposed-but-not-chosen, unknown.
- Poincare-ball reference operations in pure Python.
- C++17 performance core for `exp0`, Mobius addition, Poincare distance,
  tangent-space user updates, scoring, and ternary loss.
- Ranking evaluation helpers for chronological HR@K and NDCG@K experiments.

## Layout

```text
include/geometry_kernel/   C++17 header-only performance core
python/geometry_kernel/    Pure-Python reference and dataset utilities
tests/cpp/                 C++ tests and microbenchmarks
tests/python/              Python unit tests
```

## Build And Test

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

```bash
PYTHONPATH=python python -m unittest discover -s tests/python -v
```

PowerShell:

```powershell
$env:PYTHONPATH='python'; python -m unittest discover -s tests/python -v
```

## Python Example

```python
from geometry_kernel.datasets import load_movielens_1m
from geometry_kernel.evaluation import chronological_split

dataset = load_movielens_1m("path/to/ml-1m")
splits = chronological_split(dataset.user_sequences(min_rating=4))
```

## Numerical Contract

The C++ core projects inputs and outputs into the open Poincare ball and clips
`atanh` arguments with epsilon margins. The Euclidean limit is available with
`c <= 0`.

