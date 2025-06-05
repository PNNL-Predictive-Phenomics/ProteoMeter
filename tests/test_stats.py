from __future__ import annotations

import numpy as np
import pandas as pd
import scipy as sp

from proteometer.stats import TTestGroup, anova, pairwise_ttest


def test_pairwise_ttest():
    df = pd.DataFrame(
        {
            "A1": [1],
            "A2": [1],
            "A3": [3],
            "A4": [3],
            "A5": [2],
            "B1": [4],
            "B2": [5],
            "B3": [6],
            "B4": [6],
            "B5": [4],
        }
    )
    acols = [c for c in df.columns if "A" in c]
    bcols = [c for c in df.columns if "B" in c]
    ttest_group = TTestGroup(
        treat_group="A",
        control_group="B",
        treat_samples=acols,
        control_samples=bcols,
    )

    a = df[acols].iloc[0]
    b = df[bcols].iloc[0]

    mu_a = a.mean()
    mu_b = b.mean()

    sigma_a2 = a.std() ** 2 / len(a)  # 1/5
    sigma_b2 = b.std() ** 2 / len(b)  # 1/5
    t = (mu_a - mu_b) / np.sqrt(sigma_a2 + sigma_b2)  # -3 * np.sqrt(5 / 2)

    pval = sp.stats.t.sf(np.abs(t), len(a) + len(b) - 2) * 2
    print(t, pval)
    result = pairwise_ttest(df, [ttest_group])
    print(result)
    assert result["A/B_pval"].iloc[0] == pval
