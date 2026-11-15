"""Process-level runtime limits, applied at import time.

Why this module exists: sklearn's histogram binning and forest fitting use
joblib's ``Parallel``, which builds a worker pool (a ``multiprocessing`` queue)
even for the ``threading`` backend. In restricted environments (and under some
container security profiles) creating that queue is denied, so fitting fails
with ``PermissionError: [WinError 5]`` even though nothing is wrong with the
model or the data.

Capping the thread count to 1 keeps sklearn on its sequential path:

* ``OMP_NUM_THREADS``      - controls ``_openmp_effective_n_threads``
* ``OPENBLAS/MKL_NUM_THREADS`` - keeps linear algebra single-threaded too

It also makes training bit-for-bit reproducible: multi-threaded floating-point
reductions can change summation order, which changes the last digits of a metric.

Import this module **before** scikit-learn anywhere a model is fitted.
"""

from __future__ import annotations

import os


def limit_threads() -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


limit_threads()
