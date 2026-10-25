"""Module 08 – Exploratory data analysis and visualisation (Python / pandas + matplotlib).

Runs standalone from the project root:
    python module/08-eda-visualisierung/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.

Figures are written to this module's assets/ directory – the SAME files the
chapter (README §5–§7) displays. This script is their only source:
    ../assets/verteilung_alter.png
    ../assets/verteilung_laktat_nach_grund.png
    ../assets/streu_crp_verweildauer.png
So running this script reproduces exactly the chapter's figures.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Non-interactive backend MUST be set before pyplot is imported.
import matplotlib  # noqa: E402
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import pandas as pd  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import apply_style, PRIMARY, EVENT  # noqa: E402

apply_style()
pd.set_option("display.width", 100)
# Figures belong in the module's assets/ dir (as in every other module), not in
# code/. These are the exact files the README shows.
FIGURE_DIR = Path(__file__).parent.parent / "assets"


def prepare_data() -> pd.DataFrame:
    """Load cohort, fix gender encoding, merge with labs."""
    cohort = load_cohort()
    cohort["geschlecht"] = cohort["geschlecht"].replace({"w": "weiblich"})
    df = cohort.merge(load_labs(), on="patient_id", how="left")
    return df


def numeric_summary(df: pd.DataFrame) -> None:
    """Step 1: Print numeric and categorical overview."""
    print("=== 1) Overview: describe() ===")
    numeric_cols = ["alter", "bmi", "sofa_score", "crp_mg_l",
                    "verweildauer_tage", "laktat_mmol_l", "kreatinin_mg_dl"]
    print(df[numeric_cols].describe().round(2))

    print("\n=== 2) Frequencies: value_counts() ===")
    for col in ["aufnahmegrund", "geschlecht", "raucherstatus"]:
        print(f"\n--- {col} ---")
        print(df[col].value_counts())

    print("\n=== 3) Missing values per column ===")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string())


def detect_outliers(df: pd.DataFrame) -> None:
    """Step 2: IQR method to flag potential outliers."""
    print("\n=== 4) Outlier detection (IQR method) ===")
    for col in ["crp_mg_l", "laktat_mmol_l"]:
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((series < lower) | (series > upper)).sum()
        print(f"{col}: IQR={iqr:.2f}, bounds=[{lower:.2f}, {upper:.2f}]"
              f" → {n_out} flagged  (verify clinically before removing)")


def correlations(df: pd.DataFrame) -> None:
    """Step 3: Pearson correlation matrix for clinically relevant variables."""
    print("\n=== 5) Correlation matrix (Pearson) ===")
    cols = ["alter", "sofa_score", "crp_mg_l", "laktat_mmol_l",
            "verweildauer_tage", "verstorben_30d"]
    print(df[cols].corr().round(2))


# ── Figures ──────────────────────────────────────────────────────────────────

def plot_age_histogram(df: pd.DataFrame) -> None:
    """Histogram of age distribution, split by 30-day mortality."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for outcome, label, color in [
        (0, "Überlebt", PRIMARY),
        (1, "Verstorben", EVENT),
    ]:
        subset = df.loc[df["verstorben_30d"] == outcome, "alter"]
        ax.hist(subset, bins=20, alpha=0.65, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Alter (Jahre)")
    ax.set_ylabel("Anzahl Patient:innen")
    ax.set_title("Altersverteilung nach 30-Tage-Mortalität")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "verteilung_alter.png"
    fig.savefig(FIGURE_DIR / "verteilung_alter.png", dpi=120)   # literal path: lets the site link this code
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_lactate_boxplot(df: pd.DataFrame) -> None:
    """Boxplot: lactate by admission type, sorted by median."""
    order = (
        df.groupby("aufnahmegrund")["laktat_mmol_l"]
          .median()
          .sort_values(ascending=False)
          .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x="aufnahmegrund",
        y="laktat_mmol_l",
        order=order,
        hue="aufnahmegrund",
        palette="Set2",
        legend=False,
        ax=ax,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    ax.set_xlabel("Aufnahmegrund")
    ax.set_ylabel("Laktat (mmol/l)")
    ax.set_title("Laktatverteilung nach Aufnahmegrund")
    fig.tight_layout()
    path = FIGURE_DIR / "verteilung_laktat_nach_grund.png"
    fig.savefig(FIGURE_DIR / "verteilung_laktat_nach_grund.png", dpi=120)   # literal path: lets the site link this code
    plt.close(fig)
    print(f"Saved: {path}")


def plot_crp_los_scatter(df: pd.DataFrame) -> None:
    """Scatter plot: CRP vs length of stay, coloured by mortality, with an
    overall linear trend line – the figure the chapter (§7) discusses."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0: PRIMARY, 1: EVENT}
    for outcome, label in [(0, "Überlebt"), (1, "Verstorben")]:
        subset = df[df["verstorben_30d"] == outcome]
        ax.scatter(subset["crp_mg_l"], subset["verweildauer_tage"],
                   c=colors[outcome], label=label,
                   alpha=0.42, s=22, edgecolors="none")
    ax.set_xlabel("CRP (mg/l)")
    ax.set_ylabel("Verweildauer (Tage)")
    ax.set_title("CRP vs. Verweildauer nach 30-Tage-Mortalität")

    # Overall linear trend line (same as data/figures.py).
    mask = df["crp_mg_l"].notna() & df["verweildauer_tage"].notna()
    x_all = df.loc[mask, "crp_mg_l"].to_numpy()
    y_all = df.loc[mask, "verweildauer_tage"].to_numpy()
    coef = np.polyfit(x_all, y_all, 1)
    x_range = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(x_range, np.poly1d(coef)(x_range), color="#555555", linewidth=1.1,
            linestyle="--", alpha=0.7, label="Trend (gesamt)")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "streu_crp_verweildauer.png"
    fig.savefig(FIGURE_DIR / "streu_crp_verweildauer.png", dpi=120)   # literal path: lets the site link this code
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    df = prepare_data()

    numeric_summary(df)
    detect_outliers(df)
    correlations(df)

    print("\n=== Generating figures ===")
    plot_age_histogram(df)
    plot_lactate_boxplot(df)
    plot_crp_los_scatter(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
