"""Module 21 - Statistical test choice in practice."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from lib.helpers import load_cohort, load_labs, load_vitals  # noqa: E402


def cramers_v(table: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def main() -> None:
    cohort = load_cohort()
    labs = load_labs()
    vitals = load_vitals()
    df = cohort.merge(labs, on="patient_id", how="left")

    print("\n1) Independent continuous outcome: lactate by sepsis")
    sepsis = df.loc[df["aufnahmegrund"].eq("Sepsis"), "laktat_mmol_l"].dropna()
    other = df.loc[~df["aufnahmegrund"].eq("Sepsis"), "laktat_mmol_l"].dropna()
    print(f"Sepsis median={sepsis.median():.2f}, other median={other.median():.2f}")
    print("Welch:", stats.ttest_ind(sepsis, other, equal_var=False))
    print("Mann-Whitney:", stats.mannwhitneyu(sepsis, other, alternative="two-sided"))

    print("\n2) Paired continuous outcome: MAP day 0 vs day 3")
    wide = vitals.pivot(index="patient_id", columns="tag", values="map_mmhg").dropna(subset=[0, 3])
    diff = wide[3] - wide[0]
    print(diff.describe().round(2))
    print("Paired t:", stats.ttest_rel(wide[3], wide[0]))
    print("Wilcoxon:", stats.wilcoxon(wide[3], wide[0]))
    print("Sign test:", stats.binomtest(int((diff > 0).sum()), int((diff != 0).sum())))

    print("\n2b) Stress case: skewed differences with median exactly 0")
    # 50 small negative and 50 growing positive differences: the median
    # difference is 0, but the distribution is asymmetric. The signed-rank
    # test assumes symmetry for a statement about location and rejects anyway.
    skewed = np.r_[-np.ones(50), np.linspace(1, 50, 50)]
    print(f"median={np.median(skewed):.1f}, positive={int((skewed > 0).sum())}, "
          f"negative={int((skewed < 0).sum())}")
    print(f"Signed-rank p={stats.wilcoxon(skewed).pvalue:.6f} | "
          f"sign test p={stats.binomtest(int((skewed > 0).sum()), len(skewed)).pvalue:.3f}")

    print("\n3) More than two groups: lactate by top four admission reasons")
    top = df["aufnahmegrund"].value_counts().head(4).index
    groups = [df.loc[df["aufnahmegrund"].eq(g), "laktat_mmol_l"].dropna() for g in top]
    print(pd.DataFrame({"group": top, "n": [len(g) for g in groups], "median": [g.median() for g in groups]}))
    print("Kruskal-Wallis:", stats.kruskal(*groups))

    print("\n4) Categorical outcome: active smoking by mortality")
    tab = pd.crosstab(df["raucherstatus"].eq("aktiv"), df["verstorben_30d"])
    chi2, p, dof, expected = stats.chi2_contingency(tab, correction=False)
    print(tab)
    print("Expected counts:\n", np.round(expected, 1))
    print(f"Chi-square={chi2:.3f}, p={p:.4f}, Cramer's V={cramers_v(tab):.3f}")
    print("Fisher exact:", stats.fisher_exact(tab))


if __name__ == "__main__":
    main()
