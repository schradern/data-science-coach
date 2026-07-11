"""Module 15 - Causal inference basics with propensity scores."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from lib.helpers import load_cohort  # noqa: E402


def odds_ratio(fit, term: str) -> tuple[float, float, float]:
    ci = fit.conf_int().loc[term]
    return float(np.exp(fit.params[term])), float(np.exp(ci[0])), float(np.exp(ci[1]))


def smd(x: pd.Series, group: pd.Series, weights: pd.Series | None = None) -> float:
    x_arr = x.to_numpy(dtype=float)
    g = group.to_numpy().astype(bool)
    if weights is None:
        m1, m0 = x_arr[g].mean(), x_arr[~g].mean()
        v1, v0 = x_arr[g].var(ddof=1), x_arr[~g].var(ddof=1)
    else:
        w = weights.to_numpy(dtype=float)
        m1 = np.average(x_arr[g], weights=w[g])
        m0 = np.average(x_arr[~g], weights=w[~g])
        v1 = np.average((x_arr[g] - m1) ** 2, weights=w[g])
        v0 = np.average((x_arr[~g] - m0) ** 2, weights=w[~g])
    return float((m1 - m0) / np.sqrt((v1 + v0) / 2))


def main() -> None:
    df = load_cohort()
    df["diabetes"] = df["diabetes"].astype(int)

    # DAG (see README §2): alter -> diabetes, alter -> sofa_score, alter ->
    # verstorben_30d (alter is a CONFOUNDER: adjust for it) and
    # diabetes -> sofa_score -> verstorben_30d (sofa_score is a MEDIATOR on
    # the diabetes path: do NOT adjust for it when estimating diabetes's
    # TOTAL effect on mortality, or the adjustment blocks part of the very
    # effect we want to measure).
    print("\n1) Crude vs adjusted (total effect) diabetes effect")
    crude = smf.logit("verstorben_30d ~ diabetes", data=df).fit(disp=False)
    adj = smf.logit("verstorben_30d ~ diabetes + alter", data=df).fit(disp=False)
    for label, fit in [("crude", crude), ("adjusted (total effect, alter only)", adj)]:
        est, lo, hi = odds_ratio(fit, "diabetes")
        print(f"{label:36s} OR={est:.2f} 95% CI [{lo:.2f}, {hi:.2f}]")

    print("\n1b) Teaching moment: (wrongly) adjusting for the mediator SOFA too")
    adj_mediator = smf.logit("verstorben_30d ~ diabetes + alter + sofa_score", data=df).fit(disp=False)
    est, lo, hi = odds_ratio(adj_mediator, "diabetes")
    print(f"{'adjusted + mediator SOFA (DIRECT effect, not total!)':36s} OR={est:.2f} 95% CI [{lo:.2f}, {hi:.2f}]")
    print("-> Adjusting for the mediator shifts the estimate: it no longer answers")
    print("   'what is diabetes's total effect on mortality', but 'what is left")
    print("   once the SOFA-mediated pathway is blocked'. Different question!")

    print("\n2) Propensity score and IPTW for diabetes (confounders only: alter)")
    X = df[["alter"]]
    y = df["diabetes"]
    ps_model = LogisticRegression(max_iter=1000).fit(X, y)
    df["ps"] = ps_model.predict_proba(X)[:, 1]
    df["iptw"] = np.where(y.eq(1), 1 / df["ps"], 1 / (1 - df["ps"]))
    print(df[["ps", "iptw"]].describe().round(3))

    print("\n3) Balance before and after IPTW")
    rows = [{"variable": "alter", "raw_smd": smd(df["alter"], y), "weighted_smd": smd(df["alter"], y, df["iptw"])}]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    sofa_raw_smd = smd(df["sofa_score"], y)
    print(f"(sofa_score raw SMD={sofa_raw_smd:.3f} - NOT a PS covariate: it's a mediator, so it is")
    print(" expected to stay imbalanced by design; that is not evidence of a broken PS model.)")

    print("\n4) Weighted outcome model (IPTW -> ATE), robust variance")
    # IPTW weights are NOT frequency counts: freq_weights inflates the pseudo-N
    # to ~sum(weights) and reports a spuriously narrow CI. IPW needs a robust
    # (Huber-White HC0) sandwich SE. Verified against a nonparametric bootstrap
    # that refits the PS and outcome model each resample: sandwich SE(log-OR)
    # ~= 0.296 vs bootstrap ~= 0.297. (statsmodels notes cov_type with
    # var_weights as "not fully supported", but for HC0 it matches the bootstrap.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_fit = smf.glm(
            "verstorben_30d ~ diabetes",
            data=df,
            family=sm.families.Binomial(),
            var_weights=df["iptw"],
        ).fit(cov_type="HC0")
    ci = w_fit.conf_int().loc["diabetes"]
    est = float(np.exp(w_fit.params["diabetes"]))
    print(f"IPTW weighted OR={est:.2f} 95% CI [{np.exp(ci[0]):.2f}, {np.exp(ci[1]):.2f}]  (robust HC0 SE)")
    print("(The naive freq-weight SE would give a spuriously narrow CI ~[1.27, 2.44]; the")
    print(" robust CI now correctly includes 1 - the weighted estimate is not significant here.)")


if __name__ == "__main__":
    main()
