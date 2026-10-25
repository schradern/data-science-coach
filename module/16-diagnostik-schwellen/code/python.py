"""Module 16 - Diagnostic accuracy, threshold choice, and net benefit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend – no display needed

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import apply_style, save, PRIMARY, EVENT, SECONDARY  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def metrics_at_threshold(y_true: pd.Series, score: pd.Series, threshold: float) -> dict[str, float]:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": sens,
        "specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "LR+": sens / (1 - spec) if spec < 1 else np.inf,
        "LR-": (1 - sens) / spec if spec > 0 else np.inf,
    }


def net_benefit(y_true: np.ndarray, prob: np.ndarray, pt: float) -> float:
    """Net benefit of a model at threshold probability pt (Vickers & Elkin 2006).

    NB = TP/N - FP/N * pt/(1-pt), classifying "treat" when prob >= pt.
    """
    n = len(y_true)
    pred = prob >= pt
    tp = np.sum(pred & (y_true == 1))
    fp = np.sum(pred & (y_true == 0))
    return tp / n - fp / n * (pt / (1 - pt))


def net_benefit_all(prevalence: float, pt: float) -> float:
    """Net benefit of the 'treat everyone' strategy at threshold probability pt."""
    return prevalence - (1 - prevalence) * (pt / (1 - pt))


N_BOOTSTRAP = 2000


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the AUC (resampling patients)."""
    rng = np.random.default_rng(SEED)
    aucs = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():          # both classes needed for an AUC
            aucs.append(roc_auc_score(y[idx], score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def wilson_ci(successes: int, total: int) -> str:
    lo, hi = proportion_confint(successes, total, method="wilson")
    return f"[{lo:.3f}, {hi:.3f}]"


def main() -> None:
    df = load_cohort().merge(load_labs(), on="patient_id", how="left").dropna(subset=["laktat_mmol_l"])
    y = df["verstorben_30d"]
    score = df["laktat_mmol_l"]

    print("\n1) ROC and Youden threshold")
    auc = roc_auc_score(y, score)
    fpr, tpr, thresholds = roc_curve(y, score)
    best = int(np.argmax(tpr - fpr))
    youden = float(thresholds[best])
    auc_lo, auc_hi = bootstrap_auc_ci(y.to_numpy(), score.to_numpy())
    print(f"AUC={auc:.3f}  95% CI (bootstrap) [{auc_lo:.3f}, {auc_hi:.3f}]")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    true_pos = int(((score >= youden) & (y == 1)).sum())
    true_neg = int(((score < youden) & (y == 0)).sum())
    print(f"Youden threshold={youden:.2f}, sensitivity={tpr[best]:.3f} {wilson_ci(true_pos, n_pos)}, "
          f"specificity={1 - fpr[best]:.3f} {wilson_ci(true_neg, n_neg)}  (Wilson 95% CI)")
    print("NOTE: the Youden threshold here is chosen AND evaluated on the same data")
    print("(in-sample) – optimistic. Validate on held-out data (see modules 24/25).")

    print("\n2) Metrics at clinically simple thresholds")
    rows = [metrics_at_threshold(y, score, t) for t in [1.5, 2.0, 2.5, youden]]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n3) Prevalence")
    prevalence = float(y.mean())
    print(f"Observed event rate={prevalence:.3f}")

    # -----------------------------------------------------------------------
    # 4) Net benefit / decision-curve analysis (Vickers & Elkin 2006)
    # -----------------------------------------------------------------------
    print("\n4) Net benefit (decision-curve analysis)")
    # Youden gives ONE threshold on the raw score and ignores prevalence and the
    # relative cost of false positives vs. false negatives. A decision curve
    # encodes exactly that trade-off in the threshold probability pt: choosing pt
    # means "treat when the predicted risk exceeds pt", i.e. one false negative is
    # worth (1-pt)/pt false positives. We need predicted PROBABILITIES, so we fit
    # a simple logistic model of the outcome on laktat.
    model = smf.logit("verstorben_30d ~ laktat_mmol_l", data=df).fit(disp=False)
    prob = model.predict(df).to_numpy()
    y_arr = y.to_numpy()

    grid = np.arange(0.05, 0.51, 0.01)
    nb_model = np.array([net_benefit(y_arr, prob, pt) for pt in grid])
    nb_all = np.array([net_benefit_all(prevalence, pt) for pt in grid])
    nb_none = np.zeros_like(grid)

    print("  Net benefit at selected threshold probabilities:")
    print("   pt   NB_model  NB_all   NB_none")
    for pt in [0.10, 0.15, 0.20, 0.25, 0.30]:
        print(f"  {pt:.2f}   {net_benefit(y_arr, prob, pt):+.4f}  "
              f"{net_benefit_all(prevalence, pt):+.4f}  0.0000")
    print("  Reading: where NB_model is above BOTH treat-all and treat-none, the")
    print("  model adds decision value at that threshold probability.")

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, nb_model, color=PRIMARY, lw=2.0, label="Laktat-Modell")
    ax.plot(grid, nb_all, color=EVENT, lw=1.6, ls="--", label="Alle behandeln")
    ax.plot(grid, nb_none, color=SECONDARY, lw=1.6, ls=":", label="Niemanden behandeln")
    ax.set_xlabel("Schwellenwahrscheinlichkeit $p_t$")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision-Curve-Analyse: Laktat als Marker der 30-Tage-Mortalität")
    ax.set_ylim(-0.05, max(nb_model.max(), nb_all.max()) + 0.02)
    ax.legend(loc="upper right")
    save(fig, ASSETS / "decision_curve.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
