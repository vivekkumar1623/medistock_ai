"""Seeding policy.

One seed value flows from configuration into every stochastic component. We use
independent child generators (via ``SeedSequence``) rather than a single global
stream, because a single stream makes results depend on *call order*: adding one
extra draw anywhere silently changes every downstream number. Child streams keep
each stage reproducible in isolation.
"""

from __future__ import annotations

import numpy as np

# Stable integer tags so a stage keeps its stream even if other stages are added.
STREAM_PRODUCTS = 1_001
STREAM_SALES = 2_002
STREAM_INVENTORY = 3_003
STREAM_BATCHES = 4_004
STREAM_SPLITS = 5_005
STREAM_MODEL = 6_006


def rng_for(seed: int, stream: int) -> np.random.Generator:
    """Return the generator dedicated to ``stream`` under the master ``seed``."""
    sequence = np.random.SeedSequence([seed, stream])
    return np.random.default_rng(sequence)
