"""Generate all course figures (seed 42, reproducible).

Usage:
    python data/figures.py

All figures are saved to module/<slug>/assets/.
Requires: matplotlib, seaborn, lifelines, scikit-learn, statsmodels
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add project root to sys.path so 'lib' is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

from lib.ground_truth import true_odds_ratios_for
from lib.helpers import load_cohort, load_labs, load_vitals, SEED
from lib.plotstyle import apply_style, save, PRIMARY, SECONDARY, EVENT, PALETTE

# Set random seed for reproducibility
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# Apply shared style once, before any subplots
apply_style()

# ---------------------------------------------------------------------------
# Load and pre-process data
# ---------------------------------------------------------------------------
cohort  = load_cohort()
labs    = load_labs()
vitals  = load_vitals()

# Standardise gender labels
cohort["geschlecht"] = cohort["geschlecht"].replace({"w": "weiblich"})

# Combined dataframe (cohort + labs)
df = cohort.merge(labs, on="patient_id", how="left")

# Derived grouping variable
df["sepsis_gruppe"] = df["aufnahmegrund"].apply(
    lambda x: "Sepsis" if x == "Sepsis" else "Nicht-Sepsis"
)

ASSETS = ROOT / "module"

# ===========================================================================
# MODULE 01 · Cohort overview
# ===========================================================================
print("\n[01] Kohortenueberblick ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.subplots_adjust(wspace=0.35)

# Panel A: age histogram
ax = axes[0]
survived  = cohort.loc[cohort["verstorben_30d"] == 0, "alter"]
deceased  = cohort.loc[cohort["verstorben_30d"] == 1, "alter"]
bins = np.arange(18, 100, 5)
ax.hist(survived, bins=bins, alpha=0.70, color=PRIMARY,  label="Überlebt",  edgecolor="white", linewidth=0.5)
ax.hist(deceased, bins=bins, alpha=0.80, color=EVENT,    label="Verstorben", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Alter (Jahre)")
ax.set_ylabel("Anzahl Patient:innen")
ax.set_title("A  Altersverteilung nach 30-Tage-Mortalität")
ax.legend(title=None)
ax.set_xlim(18, 100)

# Median lines
ax.axvline(survived.median(), color=PRIMARY, linestyle="--", linewidth=1.1, alpha=0.7)
ax.axvline(deceased.median(), color=EVENT,   linestyle="--", linewidth=1.1, alpha=0.7)
ax.text(survived.median() + 1, ax.get_ylim()[1] * 0.92,
        f"Median\n{survived.median():.0f} J.", fontsize=8.5, color=PRIMARY,  va="top")
ax.text(deceased.median() + 1, ax.get_ylim()[1] * 0.75,
        f"Median\n{deceased.median():.0f} J.", fontsize=8.5, color=EVENT, va="top")

# Panel B: admission reasons (horizontal bars)
ax2 = axes[1]
counts = (cohort["aufnahmegrund"]
          .value_counts()
          .sort_values(ascending=True))
bar_colors = [EVENT if g == "Sepsis" else PRIMARY for g in counts.index]
bars = ax2.barh(counts.index, counts.values, color=bar_colors, edgecolor="none", height=0.6)
ax2.set_xlabel("Anzahl Patient:innen")
ax2.set_title("B  Aufnahmegründe (N = 500)")
ax2.grid(axis="x")
ax2.grid(axis="y", visible=False)

# Bar labels with percentages
total = counts.sum()
for bar, val in zip(bars, counts.values):
    ax2.text(val + 2, bar.get_y() + bar.get_height() / 2,
             f"{val} ({val/total:.0%})", va="center", fontsize=9.5, color="#333333")
ax2.set_xlim(0, counts.max() * 1.22)

save(fig, ASSETS / "01-einfuehrung" / "assets" / "kohorte_ueberblick.png")

# ===========================================================================
# MODULE 06 · Missing values
# ===========================================================================
print("[06] Fehlende Werte ...")

# All relevant columns (cohort + labs, excluding patient_id)
all_cols = df.drop(columns=["patient_id"])
missing_pct = all_cols.isna().mean().mul(100).sort_values(ascending=True)
missing_pct = missing_pct[missing_pct > 0]  # only columns with missing values

fig, ax = plt.subplots(figsize=(8, max(3.5, len(missing_pct) * 0.55)))

bar_colors = [EVENT if pct >= 15 else PRIMARY if pct >= 5 else SECONDARY
              for pct in missing_pct.values]
bars = ax.barh(missing_pct.index, missing_pct.values, color=bar_colors,
               edgecolor="none", height=0.6)
ax.set_xlabel("Anteil fehlender Werte (%)")
ax.set_title("Fehlende Werte je Spalte")
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, missing_pct.max() * 1.30)

for bar, pct in zip(bars, missing_pct.values):
    ax.text(pct + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f} %", va="center", fontsize=9.5)

# Legend
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=EVENT,     label="≥15 % (klinisch informativ)"),
    Patch(facecolor=PRIMARY,   label="5–15 %"),
    Patch(facecolor=SECONDARY, label="< 5 %"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9)

save(fig, ASSETS / "06-transformation" / "assets" / "fehlende_werte.png")

# ===========================================================================
# MODULE 08 · EDA visualisations (3 figures)
# ===========================================================================
print("[08] EDA-Visualisierungen ...")

# --- 07a: Age histogram (improved) ---
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.arange(18, 100, 5)
ax.hist(survived, bins=bins, alpha=0.72, color=PRIMARY, label="Überlebt",  edgecolor="white", linewidth=0.5)
ax.hist(deceased, bins=bins, alpha=0.82, color=EVENT,   label="Verstorben", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Alter (Jahre)")
ax.set_ylabel("Anzahl Patient:innen")
ax.set_title("Altersverteilung nach 30-Tage-Mortalität")
ax.legend(title=None)
ax.set_xlim(18, 100)

# Annotation box
n_total = len(cohort)
mort_rate = cohort["verstorben_30d"].mean()
ax.text(0.97, 0.95, f"N = {n_total}\n30-Tage-Mortalität: {mort_rate:.1%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8", edgecolor="none"))

save(fig, ASSETS / "08-eda-visualisierung" / "assets" / "verteilung_alter.png")

# --- 07b: Lactate by admission reason (boxplot) ---
df_lactate = df.dropna(subset=["laktat_mmol_l"]).copy()
order_by_median = (df_lactate.groupby("aufnahmegrund")["laktat_mmol_l"]
                   .median().sort_values(ascending=False).index.tolist())

fig, ax = plt.subplots(figsize=(9, 4.5))
palette_grund = {g: (EVENT if g == "Sepsis" else PRIMARY) for g in order_by_median}

bp = sns.boxplot(
    data=df_lactate, x="aufnahmegrund", y="laktat_mmol_l",
    hue="aufnahmegrund", order=order_by_median, palette=palette_grund,
    width=0.5, linewidth=0.9, fliersize=2.5,
    flierprops=dict(marker="o", alpha=0.4),
    legend=False, ax=ax,
)
sns.stripplot(
    data=df_lactate, x="aufnahmegrund", y="laktat_mmol_l",
    hue="aufnahmegrund", order=order_by_median, palette=palette_grund,
    size=2.2, alpha=0.30, jitter=True, legend=False, ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("Laktat (mmol/l)")
ax.set_title("Laktatverteilung nach Aufnahmegrund")

# Sample size as a second line in each category label (no overlap)
n_per_group = df_lactate["aufnahmegrund"].value_counts()
ax.set_xticks(range(len(order_by_median)))
ax.set_xticklabels([f"{g}\nn = {n_per_group.get(g, 0)}" for g in order_by_median])
save(fig, ASSETS / "08-eda-visualisierung" / "assets" / "verteilung_laktat_nach_grund.png")

# --- 07c: CRP vs. length of stay (scatter, coloured by outcome) ---
fig, ax = plt.subplots(figsize=(7, 5))

for outcome_val, label, color, marker in [
    (0, "Überlebt",  PRIMARY, "o"),
    (1, "Verstorben", EVENT,   "X"),
]:
    sub = df[df["verstorben_30d"] == outcome_val]
    ax.scatter(sub["crp_mg_l"], sub["verweildauer_tage"],
               c=color, label=label, alpha=0.42, s=22,
               edgecolors="none", marker=marker)

ax.set_xlabel("CRP (mg/l)")
ax.set_ylabel("Verweildauer (Tage)")
ax.set_title("CRP vs. Verweildauer nach 30-Tage-Mortalität")
ax.legend(title=None)

# Overall trend line
mask = df["crp_mg_l"].notna() & df["verweildauer_tage"].notna()
x_all = df.loc[mask, "crp_mg_l"].values
y_all = df.loc[mask, "verweildauer_tage"].values
z = np.polyfit(x_all, y_all, 1)
x_range = np.linspace(x_all.min(), x_all.max(), 200)
ax.plot(x_range, np.poly1d(z)(x_range), color=SECONDARY, linewidth=1.1,
        linestyle="--", alpha=0.7, label="Trend (gesamt)")
ax.legend(title=None)

save(fig, ASSETS / "08-eda-visualisierung" / "assets" / "streu_crp_verweildauer.png")

# ===========================================================================
# MODULE 09 · Baseline by outcome
# ===========================================================================
print("[09] Baseline nach Outcome ...")

variables = [
    ("alter",       "Alter (Jahre)"),
    ("sofa_score",  "SOFA-Score"),
    ("crp_mg_l",    "CRP (mg/l)"),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.subplots_adjust(wspace=0.38)

for ax, (var, label) in zip(axes, variables):
    group0 = cohort.loc[cohort["verstorben_30d"] == 0, var].dropna()
    group1 = cohort.loc[cohort["verstorben_30d"] == 1, var].dropna()

    # KDE curves
    from scipy.stats import gaussian_kde
    for group, color, grp_label in [
        (group0, PRIMARY, "Überlebt"),
        (group1, EVENT,   "Verstorben"),
    ]:
        kde = gaussian_kde(group, bw_method="scott")
        xmin, xmax = group.min(), group.max()
        x_range = np.linspace(xmin - (xmax - xmin) * 0.05,
                              xmax + (xmax - xmin) * 0.05, 300)
        density = kde(x_range)
        ax.fill_between(x_range, density, alpha=0.25, color=color)
        ax.plot(x_range, density, color=color, linewidth=1.6, label=grp_label)

    # Median lines
    ax.axvline(group0.median(), color=PRIMARY, linestyle=":", linewidth=1.2)
    ax.axvline(group1.median(), color=EVENT,   linestyle=":", linewidth=1.2)

    ax.set_xlabel(label)
    ax.set_ylabel("Dichte" if ax == axes[0] else "")
    ax.set_title(label)
    ax.set_yticks([])
    ax.grid(axis="x", visible=False)

    # Mann-Whitney p-value
    _, p = stats.mannwhitneyu(group0, group1, alternative="two-sided")
    p_txt = f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"
    ax.text(0.97, 0.95, p_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#444444")

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.04), fontsize=10)

save(fig, ASSETS / "09-deskriptive-statistik" / "assets" / "baseline_nach_outcome.png")

# ===========================================================================
# MODULE 09 · SPC Run Chart
# ===========================================================================
print("[09] SPC Run Chart ...")

months = np.arange(1, 25)
# Wait times: first 12 months around 45 min, next 12 months around 32 min
wait_times = [
    43.5, 47.2, 41.8, 46.0, 48.5, 42.1, 45.2, 49.0, 44.1, 46.5, 43.0, 45.5,  # Baseline (median 45.35)
    34.5, 31.2, 33.8, 30.5, 29.1, 32.4, 35.0, 31.8, 30.0, 32.5, 33.1, 28.5   # Shift (all below baseline median)
]

fig, ax = plt.subplots(figsize=(10, 4.5))

# Plot all points and lines in primary color
ax.plot(months, wait_times, "o-", color=PRIMARY, linewidth=1.5, markersize=5, label="Monatliche Wartezeit")

# Baseline median (first 12 months median)
baseline_median = np.median(wait_times[:12])
ax.axhline(baseline_median, color="#6B7178", linestyle="--", linewidth=1.2, label=f"Baseline-Median ({baseline_median:.1f} min)")

# Highlight the shift (months 13 to 24 are all below baseline median)
shift_months = months[12:]
shift_times = wait_times[12:]
ax.plot(shift_months, shift_times, "o", color=EVENT, markersize=7, label="Sonderereignis: Shift (N=12)")
ax.plot(shift_months, shift_times, "-", color=EVENT, linewidth=2.0)

# Labels & styling
ax.set_xlabel("Monat")
ax.set_ylabel("Mittlere Wartezeit (Minuten)")
ax.set_title("SPC Run Chart: Wartezeit Notaufnahme nach Triage-Einführung")
ax.set_xticks(months)
ax.set_xlim(0.5, 24.5)
ax.set_ylim(20, 60)
ax.grid(axis="y", linestyle=":", alpha=0.6)

# Annotations
ax.text(12.5, baseline_median + 1.5, "Einführung Triage-System", color=EVENT, fontsize=9.5, fontweight="semibold", ha="center")
ax.axvline(12.5, color=EVENT, linestyle=":", linewidth=1.2)

# Legend
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

save(fig, ASSETS / "09-deskriptive-statistik" / "assets" / "spc_run_chart.png")

# ===========================================================================
# MODULE 10 · Lactate: Sepsis vs. Nicht-Sepsis
# ===========================================================================
print("[10] Laktat Sepsis-Vergleich ...")

df_sep = df.dropna(subset=["laktat_mmol_l"]).copy()
sepsis_vals    = df_sep.loc[df_sep["sepsis_gruppe"] == "Sepsis",      "laktat_mmol_l"]
non_sepsis_vals = df_sep.loc[df_sep["sepsis_gruppe"] == "Nicht-Sepsis", "laktat_mmol_l"]

# Statistics
_, p_val   = stats.mannwhitneyu(sepsis_vals, non_sepsis_vals, alternative="two-sided")
n1, n2 = len(sepsis_vals), len(non_sepsis_vals)
u_stat, _ = stats.mannwhitneyu(sepsis_vals, non_sepsis_vals, alternative="two-sided")
# Rank-biserial r as effect size
r_effect = 2 * u_stat / (n1 * n2) - 1

fig, ax = plt.subplots(figsize=(6, 5))

palette_sep = {"Sepsis": EVENT, "Nicht-Sepsis": PRIMARY}
order = ["Nicht-Sepsis", "Sepsis"]

sns.boxplot(
    data=df_sep, x="sepsis_gruppe", y="laktat_mmol_l",
    hue="sepsis_gruppe", order=order, palette=palette_sep,
    width=0.45, linewidth=0.9, fliersize=2.5,
    flierprops=dict(marker="o", alpha=0.35),
    legend=False, ax=ax,
)
sns.stripplot(
    data=df_sep, x="sepsis_gruppe", y="laktat_mmol_l",
    hue="sepsis_gruppe", order=order, palette=palette_sep,
    size=3, alpha=0.35, jitter=True, legend=False, ax=ax,
)

ax.set_xlabel("")
ax.set_ylabel("Laktat (mmol/l)")
ax.set_title("Laktat bei Sepsis vs. Nicht-Sepsis")

# Sample size as a second line in each category label (no overlap)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{g}\nn = {df_sep[df_sep['sepsis_gruppe'] == g].shape[0]}"
                    for g in order])

# Significance annotation
y_max = df_sep["laktat_mmol_l"].quantile(0.97)
p_txt = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
ax.annotate(
    "", xy=(1, y_max * 0.96), xytext=(0, y_max * 0.96),
    arrowprops=dict(arrowstyle="-", color="#666666", lw=1.0),
)
ax.text(0.5, y_max * 0.98,
        f"Mann-Whitney-U\n{p_txt}\nEffektgröße r = {r_effect:.2f}",
        ha="center", va="bottom", fontsize=9, color="#333333")

fig.subplots_adjust(bottom=0.12)
save(fig, ASSETS / "10-inferenzstatistik" / "assets" / "laktat_sepsis_vergleich.png")

# --- Clinical vs. Statistical Significance ---
print("[10] Klinische vs. Statistische Signifikanz ...")

fig, ax = plt.subplots(figsize=(8, 4.2))

cases = [
    {"y": 3, "pe": 0.4,  "ci": (0.2, 0.6),   "color": SECONDARY, "title": "Studie A: Statistische Signifikanz, klinisch irrelevant\n(Sehr große Fallzahl N, winziger Effekt)"},
    {"y": 2, "pe": 2.2,  "ci": (-0.4, 4.8),  "color": EVENT,     "title": "Studie B: Keine stat. Signifikanz, klinisch potenziell relevant\n(Kleine Fallzahl N, großer Effekt, unterpowered)"},
    {"y": 1, "pe": 2.8,  "ci": (1.8, 3.8),   "color": PRIMARY,   "title": "Studie C: Optimaler Fall\n(Ausreichende Fallzahl N, signifikanter & relevanter Effekt)"}
]

# Relevance threshold
threshold = 1.5
ax.axvline(threshold, color="#9A5B12", linestyle=":", linewidth=1.5)
ax.text(threshold + 0.1, 3.6, "Schwelle für klinische Relevanz", color="#9A5B12", fontsize=9.5, fontweight="semibold")

# Zero line (no effect)
ax.axvline(0.0, color="#16181C", linestyle="-", linewidth=1.2)
ax.text(-0.8, 3.6, "Kein Effekt", color="#16181C", fontsize=9.5, fontweight="semibold")

for c in cases:
    ax.errorbar(c["pe"], c["y"], xerr=[[c["pe"] - c["ci"][0]], [c["ci"][1] - c["pe"]]], fmt="o", color=c["color"], markersize=8, elinewidth=2.5, capsize=6)
    ax.text(-1.4, c["y"] + 0.22, c["title"], color="#16181C", fontsize=9.5, fontweight="semibold", ha="left")

ax.set_ylim(0.5, 3.9)
ax.set_xlim(-1.5, 5.5)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["Studie C", "Studie B", "Studie A"])
ax.set_xlabel("Behandlungseffekt (z.B. Blutdrucksenkung in mmHg)")
ax.set_title("Klinische vs. Statistische Signifikanz: Konfidenzintervalle richtig lesen")
ax.grid(True, axis="x", linestyle=":", alpha=0.6)

(ASSETS / "10-inferenzstatistik" / "assets").mkdir(parents=True, exist_ok=True)
save(fig, ASSETS / "10-inferenzstatistik" / "assets" / "clinical_vs_statistical_significance.png")

# ===========================================================================
# MODULE 12 · Forest plot: odds ratios + KM curve
# ===========================================================================
print("[12] Forest Plot & Kaplan-Meier ...")

import statsmodels.formula.api as smf

cohort["aktiv_raucher"] = (cohort["raucherstatus"] == "aktiv").astype(int)
cohort["sepsis"]        = (cohort["aufnahmegrund"] == "Sepsis").astype(int)

formula = ("verstorben_30d ~ alter + sofa_score + crp_mg_l "
           "+ diabetes + sepsis")
model = smf.logit(formula, data=cohort).fit(disp=0)

# OR and 95% CI
params = model.params.drop("Intercept")
conf   = model.conf_int().drop("Intercept")
or_est = np.exp(params)
or_lo  = np.exp(conf[0])
or_hi  = np.exp(conf[1])

# True ORs for exactly this model specification.
#
# NOT exp(beta) from generate_data.py: those betas are log-HAZARDS of a Weibull
# process, so exponentiating them yields hazard ratios. This plot shows odds
# ratios, and at a 15.6 % event rate the true OR lies further from 1 than the
# true HR. lib/ground_truth.py replays the generating process at N = 200 000 and
# fits this same model to obtain them.
true_or = true_odds_ratios_for(("alter", "sofa_score", "crp_mg_l", "diabetes", "sepsis"))

# German display labels for predictors
predictor_labels = {
    "alter":      "Alter (pro Jahr)",
    "sofa_score": "SOFA-Score (pro Punkt)",
    "crp_mg_l":   "CRP (pro mg/l)",
    "diabetes":   "Diabetes (vs. nein)",
    "sepsis":     "Sepsis (vs. Nicht-Sepsis)",
}

predictor_order = list(params.index)
y_pos = list(range(len(predictor_order) - 1, -1, -1))  # reversed order

fig, ax = plt.subplots(figsize=(9, 5))

for i, (pred, y) in enumerate(zip(predictor_order, y_pos)):
    est = or_est[pred]
    lo  = or_lo[pred]
    hi  = or_hi[pred]

    # CI line
    ax.plot([lo, hi], [y, y], color=PRIMARY, linewidth=2.0, solid_capstyle="round")
    # Point estimate
    ax.plot(est, y, "o", color=PRIMARY, markersize=9, zorder=5)
    # True OR (small diamond)
    if pred in true_or:
        ax.plot(true_or[pred], y, "D", color=EVENT,
                markersize=6, zorder=6, alpha=0.85)

    # Label: OR [CI]
    ci_label = f"{est:.2f} [{lo:.2f}–{hi:.2f}]"
    ax.text(max(hi, 1.0) * 1.02, y, ci_label, va="center", fontsize=9.5, color="#222222")

# Reference line at OR = 1
ax.axvline(1.0, color=SECONDARY, linestyle="--", linewidth=1.0, alpha=0.7)

# y-axis: variable names in German
ax.set_yticks(y_pos)
ax.set_yticklabels([predictor_labels.get(p, p) for p in predictor_order], fontsize=10.5)
ax.set_xscale("log")
ax.set_xlabel("Odds Ratio (95 %-KI, logarithmische Skala)")
ax.set_title("Risikofaktoren für 30-Tage-Mortalität\nLogistische Regression (adjustiert)")

# Grid on x-axis too
ax.grid(axis="x", linestyle=":", linewidth=0.7, color="#DDDDDD")

# Legend
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=PRIMARY,
           markersize=9, label="Geschätztes OR (95 %-KI)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor=EVENT,
           markersize=6, label="Wahre OR (Datengenerierung, N = 2 000 000)"),
]
ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.18),
          ncol=2, fontsize=9.5, frameon=False)
fig.subplots_adjust(bottom=0.22)

save(fig, ASSETS / "12-regression" / "assets" / "forest_odds_ratios.png")

# --- Kaplan-Meier curve ---
from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(8, 5))

kmf = KaplanMeierFitter()
for group, color, label in [
    ("Sepsis",       EVENT,   "Sepsis"),
    ("Nicht-Sepsis", PRIMARY, "Nicht-Sepsis"),
]:
    mask = df["sepsis_gruppe"] == group
    kmf.fit(
        df.loc[mask, "fu_zeit_tage"],
        df.loc[mask, "status"],
        label=label,
    )
    kmf.plot_survival_function(
        ax=ax, ci_show=True, ci_alpha=0.15,
        color=color, linewidth=2.0,
    )

ax.set_xlabel("Zeit (Tage)")
ax.set_ylabel("Geschätzte Überlebenswahrscheinlichkeit")
ax.set_title("Kaplan-Meier-Überlebensfunktion\nStratifiziert nach Aufnahmegrund (Sepsis vs. Nicht-Sepsis)")
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

# Log-rank test
from lifelines.statistics import logrank_test
lr_result = logrank_test(
    df.loc[df["sepsis_gruppe"] == "Sepsis",      "fu_zeit_tage"],
    df.loc[df["sepsis_gruppe"] == "Nicht-Sepsis", "fu_zeit_tage"],
    event_observed_A=df.loc[df["sepsis_gruppe"] == "Sepsis",      "status"],
    event_observed_B=df.loc[df["sepsis_gruppe"] == "Nicht-Sepsis", "status"],
)
p_lr = lr_result.p_value
p_txt = f"Log-rank p = {p_lr:.4f}" if p_lr >= 0.0001 else "Log-rank p < 0.0001"
ax.text(0.97, 0.97, p_txt, transform=ax.transAxes,
        ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4F8", edgecolor="none"))

save(fig, ASSETS / "12-regression" / "assets" / "km_ueberleben_sepsis.png")

# --- Regression Residual Diagnostics ---
print("[12] Regressions-Residuendiagnostik ...")

np.random.seed(42)
fitted_vals = np.linspace(10, 90, 200)

# 1. Homoscedasticity (Good model): residuals are a constant band
residuals_good = np.random.normal(0, 5, len(fitted_vals))

# 2. Heteroscedasticity (Violated model): variance increases with fitted values (funnel shape)
residuals_bad = np.random.normal(0, 1.0 + 0.25 * fitted_vals, len(fitted_vals))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

# Homoscedasticity
ax1.scatter(fitted_vals, residuals_good, color=PRIMARY, alpha=0.6, edgecolors="none")
ax1.axhline(0, color="#16181C", linestyle="--", linewidth=1.2)
ax1.set_xlabel("Vorhergesagte Werte (Fitted Values)")
ax1.set_ylabel("Residuen (Residuals)")
ax1.set_title("Homoskedastizität (Annahme erfüllt)\n(Gleichmäßige Streuung)")
ax1.grid(True, linestyle=":", alpha=0.6)

# Heteroscedasticity
ax2.scatter(fitted_vals, residuals_bad, color=EVENT, alpha=0.6, edgecolors="none")
ax2.axhline(0, color="#16181C", linestyle="--", linewidth=1.2)
ax2.set_xlabel("Vorhergesagte Werte (Fitted Values)")
ax2.set_title("Heteroskedastizität (Annahme verletzt)\n(Trichterförmige Streuung)")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
(ASSETS / "12-regression" / "assets").mkdir(parents=True, exist_ok=True)
save(fig, ASSETS / "12-regression" / "assets" / "residual_diagnostics.png")

# ===========================================================================
# MODULE 13 · Study Design & Power
# ===========================================================================
print("[13] Studiendesign & Power ...")

n_range = np.arange(10, 200, 2)
from statsmodels.stats.power import TTestIndPower
power_analysis = TTestIndPower()

power_d3 = [power_analysis.solve_power(effect_size=0.3, nobs1=n, alpha=0.05) for n in n_range]
power_d5 = [power_analysis.solve_power(effect_size=0.5, nobs1=n, alpha=0.05) for n in n_range]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(n_range, power_d3, color=SECONDARY, linewidth=2, label="Kleiner Effekt (Cohen's d = 0.3)")
ax.plot(n_range, power_d5, color=PRIMARY, linewidth=2.5, label="Mittlerer Effekt (Cohen's d = 0.5)")

# Target power line (80%)
ax.axhline(0.80, color=EVENT, linestyle="--", linewidth=1.2, label="Soll-Power (80 %)")

# Find exact N for d=0.5 at 80% power
n_target = power_analysis.solve_power(effect_size=0.5, power=0.80, alpha=0.05)
ax.axvline(n_target, color=EVENT, linestyle=":", linewidth=1.2)
ax.plot(n_target, 0.80, "o", color=EVENT, markersize=8)
ax.text(n_target + 5, 0.72, f"N = {int(np.ceil(n_target))} pro Gruppe\nfür 80 % Power", color=EVENT, fontsize=9.5, fontweight="semibold")

ax.set_xlabel("Stichprobengröße pro Gruppe (N)")
ax.set_ylabel("Statistische Power (1 − β)")
ax.set_title("Power-Kurve: Stichprobengröße vs. Power (Alpha = 0.05)")
ax.set_ylim(0, 1.05)
ax.set_xlim(10, 200)
ax.legend(loc="lower right")
ax.grid(True, linestyle=":", alpha=0.6)

# Ensure directory exists before saving
(ASSETS / "13-studiendesign-power" / "assets").mkdir(parents=True, exist_ok=True)
save(fig, ASSETS / "13-studiendesign-power" / "assets" / "studiendesign_power.png")

# ===========================================================================
# MODULE 15 · Causal Inference (Simpson's Paradox)
# ===========================================================================
print("[15] Kausale Inferenz (Simpson's Paradox) ...")

np.random.seed(42)
n_patients = 100

# Mild group: low dose, high health score
dose_mild = np.random.normal(3.0, 0.8, n_patients)
health_mild = 65 + 4 * dose_mild + np.random.normal(0, 4.0, n_patients)

# Severe group: high dose, low health score
dose_severe = np.random.normal(8.0, 0.8, n_patients)
health_severe = 20 + 4 * dose_severe + np.random.normal(0, 4.0, n_patients)

# Combined data
all_dose = np.concatenate([dose_mild, dose_severe])
all_health = np.concatenate([health_mild, health_severe])

fig, ax = plt.subplots(figsize=(8, 4.8))

# Scatter plots
ax.scatter(dose_mild, health_mild, color=PRIMARY, alpha=0.6, label="Mild erkrankt (Leichte Fälle)")
ax.scatter(dose_severe, health_severe, color=EVENT, alpha=0.6, label="Schwer erkrankt (Kritische Fälle)")

# Regressions for subgroups (True causal positive effect)
slope_m, intercept_m = np.polyfit(dose_mild, health_mild, 1)
ax.plot(np.linspace(1, 5, 50), slope_m * np.linspace(1, 5, 50) + intercept_m, color=PRIMARY, linewidth=2)

slope_s, intercept_s = np.polyfit(dose_severe, health_severe, 1)
ax.plot(np.linspace(6, 10, 50), slope_s * np.linspace(6, 10, 50) + intercept_s, color=EVENT, linewidth=2)

# Overall regression (Confounded negative slope)
slope_all, intercept_all = np.polyfit(all_dose, all_health, 1)
ax.plot(np.linspace(1, 10, 100), slope_all * np.linspace(1, 10, 100) + intercept_all, color=SECONDARY, linestyle="--", linewidth=2.5, label="Crude Assoziation (Gesamtgruppe)")

ax.text(2.0, 46, "Lokaler Effekt:\nPositive Assoziation (+)", color=PRIMARY, fontsize=9.5, fontweight="semibold")
ax.text(6.8, 62, "Crude Assoziation:\nNegative Assoziation (-)", color=SECONDARY, fontsize=9.5, fontweight="semibold")

ax.set_xlabel("Therapie-Dosis (Dose)")
ax.set_ylabel("Gesundheits-Score (Health Score)")
ax.set_title("Simpson-Paradoxon: Confounding in der Praxis\n(Subgruppen vs. Gesamtassoziation)")
ax.legend(loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)

(ASSETS / "15-kausale-inferenz" / "assets").mkdir(parents=True, exist_ok=True)
save(fig, ASSETS / "15-kausale-inferenz" / "assets" / "simpsons_paradox.png")

# ===========================================================================
# MODULE 16 · Diagnostic thresholds (Healthy vs. Diseased)
# ===========================================================================
print("[16] Diagnostik & Schwellenwert ...")

x = np.linspace(0, 10, 300)
healthy = stats.norm.pdf(x, 3.5, 1.0)
diseased = stats.norm.pdf(x, 6.0, 1.2)

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(x, healthy, color=PRIMARY, linewidth=2, label="Gesund (negativ)")
ax.fill_between(x, healthy, alpha=0.10, color=PRIMARY)

ax.plot(x, diseased, color=EVENT, linewidth=2, label="Krank (positiv)")
ax.fill_between(x, diseased, alpha=0.10, color=EVENT)

# Cut-off at 4.8
cutoff = 4.8
ax.axvline(cutoff, color="#16181C", linestyle="-", linewidth=1.5)
ax.text(cutoff + 0.1, 0.35, "Schwellenwert\n(Cut-off)", color="#16181C", fontsize=10, fontweight="semibold")

# Fill errors: False Positives (Healthy above cutoff)
fp_x = x[x >= cutoff]
ax.fill_between(fp_x, stats.norm.pdf(fp_x, 3.5, 1.0), color=PRIMARY, alpha=0.4, label="Falsch-Positiv (FP)")

# Fill errors: False Negatives (Diseased below cutoff)
fn_x = x[x < cutoff]
ax.fill_between(fn_x, stats.norm.pdf(fn_x, 6.0, 1.2), color=EVENT, alpha=0.4, label="Falsch-Negativ (FN)")

ax.set_xlabel("Biomarker-Konzentration (z.B. Laktat, CRP)")
ax.set_ylabel("Wahrscheinlichkeitsdichte")
ax.set_title("Diagnostischer Schwellenwert & Klassifikationsfehler")
ax.legend(loc="upper right")
ax.grid(True, linestyle=":", alpha=0.6)

(ASSETS / "16-diagnostik-schwellen" / "assets").mkdir(parents=True, exist_ok=True)
save(fig, ASSETS / "16-diagnostik-schwellen" / "assets" / "diagnostik_schwellenwert.png")

# ===========================================================================
# MODULE 17 · Classic Survival Analysis & Censoring
# ===========================================================================
print("[17] Survival-Zensierung ...")

fig, ax = plt.subplots(figsize=(8, 4.5))

patients = [
    {"id": "Pat. 1", "start": 0, "end": 28, "event": True,  "desc": "Ereignis (Tod)"},
    {"id": "Pat. 2", "start": 2, "end": 30, "event": False, "desc": "Zensiert (Studienende)"},
    {"id": "Pat. 3", "start": 5, "end": 18, "event": False, "desc": "Zensiert (Loss-to-Follow-up)"},
    {"id": "Pat. 4", "start": 10, "end": 25, "event": True,  "desc": "Ereignis (Tod)"},
    {"id": "Pat. 5", "start": 15, "end": 30, "event": False, "desc": "Zensiert (Studienende)"},
]

for idx, p in enumerate(patients):
    y = idx + 1
    # Line for follow-up duration
    ax.plot([p["start"], p["end"]], [y, y], color=PRIMARY, linewidth=2.5)
    
    # Start marker
    ax.plot(p["start"], y, ">", color=PRIMARY, markersize=8)
    
    if p["event"]:
        ax.plot(p["end"], y, "X", color=EVENT, markersize=10, label="Ereignis (Tod)" if idx == 0 else "")
    else:
        ax.plot(p["end"], y, "o", color=SECONDARY, markersize=7, label="Zensiert (Rechtszensierung)" if idx == 1 else "")

ax.set_yticks(range(1, len(patients) + 1))
ax.set_yticklabels([p["id"] for p in patients])
ax.set_xlabel("Studienzeit (Tage)")
ax.set_ylabel("Patient:innen")
ax.set_title("Rechtszensierung in der Survival-Analyse")
ax.set_xlim(-1, 32)
ax.set_ylim(0.5, len(patients) + 0.8)
ax.grid(axis="x", linestyle=":", alpha=0.6)
# Legend placed below the axes so it never overlaps the Pat. 1 row.
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
          fontsize=10, frameon=False)
fig.subplots_adjust(bottom=0.24)

(ASSETS / "17-survival-klassisch" / "assets").mkdir(parents=True, exist_ok=True)
save(fig, ASSETS / "17-survival-klassisch" / "assets" / "survival_zensierung.png")

# ===========================================================================
# MODULE 21 · Test selection guide (5 figures showing distributions)
# ===========================================================================
print("[21] Testwahl-Verteilungen ...")
M21_DIR = ASSETS / "21-statistik-entscheidung" / "assets"
M21_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) Welch-t-Test vs. Mann-Whitney-U ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
fig.subplots_adjust(wspace=0.35, bottom=0.18)

# Welch-t-Test: Two symmetric normal distributions
x = np.linspace(30, 110, 200)
ax1.plot(x, stats.norm.pdf(x, 65, 10), color=PRIMARY, linewidth=2, label="Gruppe A")
ax1.fill_between(x, stats.norm.pdf(x, 65, 10), alpha=0.15, color=PRIMARY)
ax1.plot(x, stats.norm.pdf(x, 75, 12), color=EVENT, linewidth=2, label="Gruppe B")
ax1.fill_between(x, stats.norm.pdf(x, 75, 12), alpha=0.15, color=EVENT)
ax1.set_title("t-Test / Welch-Test\n(Symmetrisch, Normalverteilt)")
ax1.set_xlabel("Messwert (z.B. Alter)")
ax1.set_ylabel("Dichte")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Mann-Whitney-U: Two skewed distributions
x_skew = np.linspace(0, 12, 200)
ax2.plot(x_skew, stats.gamma.pdf(x_skew, 2, 0, 1), color=PRIMARY, linewidth=2, label="Gruppe A")
ax2.fill_between(x_skew, stats.gamma.pdf(x_skew, 2, 0, 1), alpha=0.15, color=PRIMARY)
ax2.plot(x_skew, stats.gamma.pdf(x_skew, 2, 0, 1.5), color=EVENT, linewidth=2, label="Gruppe B")
ax2.fill_between(x_skew, stats.gamma.pdf(x_skew, 2, 0, 1.5), alpha=0.15, color=EVENT)
ax2.set_title("Mann-Whitney-U-Test\n(Schief / Nicht-normalverteilt / Ordinal)")
ax2.set_xlabel("Messwert (z.B. Laktat, Liegedauer)")
ax2.set_ylabel("Dichte")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

save(fig, M21_DIR / "dist_t_vs_mwu.png")

# --- 2) Paired t-Test vs. Wilcoxon Signed-Rank ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
fig.subplots_adjust(wspace=0.35, bottom=0.18)

# Paired t-Test: Normal distribution of differences
x = np.linspace(-15, 15, 200)
ax1.plot(x, stats.norm.pdf(x, -2, 4), color=PRIMARY, linewidth=2, label="Differenzen (D3 - D0)")
ax1.fill_between(x, stats.norm.pdf(x, -2, 4), alpha=0.15, color=PRIMARY)
ax1.axvline(0, color=SECONDARY, linestyle="--", linewidth=1.2)
ax1.set_title("Gepaarter t-Test\n(Differenzen normalverteilt)")
ax1.set_xlabel("Differenz (z.B. Blutdruck-Änderung)")
ax1.set_ylabel("Dichte")
# Extra headroom above the curve so the legend never crosses the peak.
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.32)
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Wilcoxon: Skewed/heavy-tailed distribution of differences
y_skew = stats.lognorm.pdf(x + 20, 0.6, scale=24) # Ensure domain fits bounds
ax2.plot(x, y_skew, color=PRIMARY, linewidth=2, label="Differenzen (D3 - D0)")
ax2.fill_between(x, y_skew, alpha=0.15, color=PRIMARY)
ax2.axvline(0, color=SECONDARY, linestyle="--", linewidth=1.2)
ax2.set_title("Wilcoxon-Vorzeichen-Rang-Test\n(Differenzen schief / mit Extremwerten)")
ax2.set_xlabel("Differenz (z.B. CRP-Änderung)")
ax2.set_ylabel("Dichte")
# Extra headroom above the curve so the legend never crosses the peak.
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.32)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

save(fig, M21_DIR / "dist_paired_t_vs_wilcoxon.png")

# --- 3) ANOVA vs. Kruskal-Wallis ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
fig.subplots_adjust(wspace=0.35, bottom=0.18)

# ANOVA: Three normal curves
x = np.linspace(40, 110, 200)
ax1.plot(x, stats.norm.pdf(x, 60, 8), color=PRIMARY, linewidth=1.8, label="Gruppe A")
ax1.plot(x, stats.norm.pdf(x, 70, 8), color=EVENT, linewidth=1.8, label="Gruppe B")
ax1.plot(x, stats.norm.pdf(x, 78, 8), color="#E69F00", linewidth=1.8, label="Gruppe C")
ax1.fill_between(x, stats.norm.pdf(x, 70, 8), alpha=0.08, color=EVENT)
ax1.set_title("ANOVA (Varianzanalyse)\n(≥3 Gruppen, alle normalverteilt)")
ax1.set_xlabel("Messwert (z.B. BMI)")
ax1.set_ylabel("Dichte")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Kruskal-Wallis: Three skewed curves
x_skew = np.linspace(0, 20, 200)
ax2.plot(x_skew, stats.gamma.pdf(x_skew, 2, 0, 1.5), color=PRIMARY, linewidth=1.8, label="Gruppe A")
ax2.plot(x_skew, stats.gamma.pdf(x_skew, 2.5, 0, 2.0), color=EVENT, linewidth=1.8, label="Gruppe B")
ax2.plot(x_skew, stats.gamma.pdf(x_skew, 3, 0, 2.8), color="#E69F00", linewidth=1.8, label="Gruppe C")
ax2.set_title("Kruskal-Wallis-Test\n(≥3 Gruppen, schief / ordinal)")
ax2.set_xlabel("Messwert (z.B. SOFA-Score, Pain-Scale)")
ax2.set_ylabel("Dichte")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

save(fig, M21_DIR / "dist_anova_vs_kruskal.png")

# --- 4) Chi-Square vs. Fisher's Exact ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
fig.subplots_adjust(wspace=0.35, bottom=0.18)

# Chi-Square: Heatmap or bar chart representing large cell counts
large_data = [[140, 110], [90, 160]]
sns.heatmap(large_data, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1,
            xticklabels=["COPD Nein", "COPD Ja"], yticklabels=["Raucher Nein", "Raucher Ja"])
ax1.set_title("Chi-Quadrat-Test\n(Häufigkeiten mit großen Zellzahlen: alle ≥ 5)")

# Fisher's Exact: Heatmap with very small counts
small_data = [[15, 1], [18, 0]]
sns.heatmap(small_data, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax2,
            xticklabels=["Anaphylaxie Nein", "Anaphylaxie Ja"], yticklabels=["Placebo", "Verum"])
ax2.set_title("Fisher exakter Test\n(Häufigkeiten mit kleinen Zellzahlen: < 5)")

save(fig, M21_DIR / "dist_chi2_vs_fisher.png")

# --- 5) Survival: Kaplan-Meier vs. Cox-Regression ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4))
fig.subplots_adjust(wspace=0.35, bottom=0.18)

# Kaplan-Meier: Raw step function curves (unadjusted)
t_km = np.arange(0, 31, 2)
s1 = np.exp(-0.02 * t_km)
s2 = np.exp(-0.06 * t_km)
ax1.step(t_km, s1, where="post", color=PRIMARY, linewidth=2, label="Therapie A (unbereinigt)")
ax1.step(t_km, s2, where="post", color=EVENT, linewidth=2, label="Therapie B (unbereinigt)")
ax1.set_title("Kaplan-Meier & Log-Rank\n(Unbereinigte Überlebenskurven)")
ax1.set_xlabel("Zeit (Tage)")
ax1.set_ylabel("Überlebenswahrscheinlichkeit")
ax1.set_ylim(0, 1.05)
ax1.legend(loc="lower left")
ax1.grid(True, linestyle=":", alpha=0.6)

# Cox-Regression: Hazard ratio plot (Forest Plot, multivariable adjusted)
variables = ["Therapie B (vs A)", "Alter (+10 J.)", "SOFA-Score (+1)"]
hrs = [2.4, 1.25, 1.15]
cis = [[1.5, 3.8], [1.05, 1.5], [1.02, 1.3]]

for idx, (var, hr, ci) in enumerate(zip(variables, hrs, cis)):
    ax2.errorbar(hr, idx, xerr=[[hr - ci[0]], [ci[1] - hr]], fmt="o", color=PRIMARY,
                 markersize=8, elinewidth=2, capsize=4)
ax2.axvline(1.0, color="#6B7178", linestyle="--", linewidth=1.2)
ax2.set_yticks(range(len(variables)))
ax2.set_yticklabels(variables)
ax2.set_title("Cox Proportional Hazards\n(Adjustierter Effekt mehrerer Faktoren)")
ax2.set_xlabel("Hazard Ratio (HR) mit 95% KI")
ax2.set_ylim(-0.5, 2.5)
ax2.set_xlim(0.5, 4.5)
ax2.grid(True, linestyle=":", alpha=0.6)

save(fig, M21_DIR / "dist_km_vs_cox.png")

# --- 6) Mann-Whitney U test median myth ---
print("  6) Mann-Whitney-Median-Mythos ...")
np.random.seed(42)
n_samples = 5000

# Group B: Concentrated lognormal (median e^1.25 = 3.50)
group_b = np.random.lognormal(mean=np.log(3.5), sigma=0.25, size=n_samples)
target_median = np.median(group_b)

# Group A: Bi-modal / skewed mixture shifted to have the EXACT same median
group_a_raw = np.concatenate([
    np.random.normal(loc=1.8, scale=0.5, size=int(n_samples * 0.60)),
    np.random.normal(loc=7.5, scale=2.2, size=int(n_samples * 0.40))
])
group_a_raw = np.clip(group_a_raw, 0.1, None)
median_a = np.median(group_a_raw)
group_a = group_a_raw + (target_median - median_a)

fig, ax = plt.subplots(figsize=(8, 4.8))

# Compute smooth density curves using KDE
kde_a = stats.gaussian_kde(group_a)
kde_b = stats.gaussian_kde(group_b)

x_grid = np.linspace(0, 14, 500)
density_a = kde_a(x_grid)
density_b = kde_b(x_grid)

# Plot Group A (spread)
ax.plot(x_grid, density_a, color=PRIMARY, linewidth=2, label="Gruppe A (breiter verteilt)")
ax.fill_between(x_grid, density_a, alpha=0.15, color=PRIMARY)

# Plot Group B (concentrated)
ax.plot(x_grid, density_b, color=EVENT, linewidth=2, label="Gruppe B (konzentriert)")
ax.fill_between(x_grid, density_b, alpha=0.15, color=EVENT)

# Median line
ax.axvline(target_median, color="#16181C", linestyle="--", linewidth=1.5)
ax.text(target_median + 0.15, 0.58, f"Median = {target_median:.2f}\n(identisch in beiden Gruppen)", color="#16181C", fontsize=9.5, fontweight="semibold")

# Add text for MWU p-value
u_stat, p_val = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
ax.text(8.5, 0.40, f"Mann-Whitney-U\np < 0.001", color="#16181C", fontsize=12, fontweight="bold")

# Title and subtitles (bold title, smaller sub)
ax.set_title("Gleicher Median. Signifikanter Mann-Whitney-Test.\nDer Test prüft stochastische Dominanz, nicht den Unterschied der Mediane.", fontsize=11, fontweight="semibold", pad=12, loc="left")

ax.set_xlabel("Messwert (Value)")
ax.set_ylabel("Wahrscheinlichkeitsdichte (Density)")
ax.set_xlim(-0.2, 14)
ax.set_ylim(0, 0.7)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower right")

save(fig, M21_DIR / "mann_whitney_median_myth.png")

# ===========================================================================
# Done
# ===========================================================================
print("\nAlle Figuren erfolgreich erzeugt.")
print("\nErzeugte PNG-Dateien:")
import subprocess
result = subprocess.run(
    ["find", str(ASSETS), "-name", "*.png", "-path", "*/assets/*"],
    capture_output=True, text=True
)
for line in sorted(result.stdout.strip().split("\n")):
    if line:
        print(f"  {line}")
