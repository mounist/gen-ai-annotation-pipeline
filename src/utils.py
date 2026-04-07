"""
Shared utilities for the annotation pipeline.
"""

import numpy as np
import pandas as pd
from collections import Counter

CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]
ANNOTATOR_COLS = ["annotator_1", "annotator_2", "annotator_3", "annotator_4", "annotator_5"]


def compute_consensus(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Majority vote among 5 human annotators with random tie-breaking.

    Args:
        df: DataFrame with annotator columns.
        seed: Random seed for reproducible tie-breaking.

    Returns:
        Series of consensus labels aligned with df.index.
    """
    rng = np.random.default_rng(seed)
    consensus = []
    for _, row in df.iterrows():
        votes = [row[c] for c in ANNOTATOR_COLS]
        counter = Counter(votes)
        max_count = counter.most_common(1)[0][1]
        top = [cat for cat, cnt in counter.items() if cnt == max_count]
        consensus.append(rng.choice(top) if len(top) > 1 else top[0])
    return pd.Series(consensus, index=df.index)
