"""Generate the synthetic course datasets – reproducible (seed 42).

Run:
    ./.venv/bin/python data/generate_data.py

Creates, in this folder:
    kohorte.csv       one row per patient (master / admission data, outcome,
                       survival time + event status)
    vitalwerte.csv    several measurements per patient (long format)
    labor.csv         laboratory values per patient (wide, with MAR missing
                       values in laktat_mmol_l)
    notizen.csv       synthetic free-text clinical notes (for NLP)
    konkurrenz_risiken.csv  coherent competing-risks (death vs. discharge) +
                       time-varying vasopressor exposure – Modul 20
    kohorte_de.csv    the cohort in "German hospital export" format
                       (sep=";", encoding="latin-1", decimal comma) – Modul 04
    einschluss.xlsx   a 3-sheet inclusion-list workbook (Kohorte/Labor/Klinik)
                       – Modul 04
    beispiel.json     a small flat JSON array (REDCap-like export) – Modul 04
    fhir_bundle.json  a minimal, fictitious FHIR R4 Bundle – Modul 04

The data is entirely FICTITIOUS – no real patient data. It mimics an
internal-medicine / intensive-care cohort so the lessons stay clinically
plausible. Code is English; the column names and categories are German (the
dataset schema of a German hospital).

Ground truth (by construction – used across the course to check learners'
estimates against the true data-generating process):
  - `sofa_score` is an intermediate / mediator on the diabetes -> death and
    sepsis -> death paths (comorbidity/admission reason raise SOFA, which
    raises the hazard). `alter` is a confounder, not a mediator.
  - 30-day mortality follows a Weibull time-to-event process (see
    `generate_cohort`), NOT a plain logistic draw independent of time.
  - `laktat_mmol_l` is missing-at-random (MAR): missingness depends on
    `sofa_score` (sicker patients are measured more often -> missing less),
    not on the value itself and not on `alter`.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N = 500
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# An .xlsx is a zip archive, and both the archive headers and docProps/core.xml
# would otherwise record the moment the file was written. A fixed instant keeps
# `generate_data.py` byte-for-byte reproducible; the date itself is arbitrary.
XLSX_EPOCH = datetime(2024, 1, 1, 0, 0, 0)
sys.path.insert(0, str(ROOT))

from lib.ground_truth import (  # noqa: E402
    AGE_MAX,
    AGE_MEAN,
    AGE_MIN,
    AGE_SD,
    BGA_DEAD,
    BGA_INTERCEPT,
    BGA_MISS_B0,
    BGA_MISS_DEAD,
    BGA_NOISE_SD,
    BGA_SOFA,
    CRP_MAX,
    CRP_MIN,
    CRP_SCALE,
    CRP_SHAPE,
    DIAB_AGE_REF,
    DIAB_B0,
    DIAB_MAX,
    DIAB_MIN,
    DIAB_SLOPE,
    REASON_P,
    REASONS,
    SMOKING,
    SMOKING_P,
    SOFA_MAX,
    SOFA_MIN,
    SOFA_POISSON_MEAN,
    SOFA_SEPSIS_BONUS,
)

# The Weibull survival constants and the log-hazard coefficients live in
# lib/ground_truth.py, which replays this process to obtain the true effects.
# They are imported, never restated: a second copy is a second chance to drift.
from lib.ground_truth import (  # noqa: E402
    ADMIN_HORIZON_DAYS,
    BASELINE_SCALE,
    CENSOR_HI,
    CENSOR_LO,
    LOST_TO_FU_RATE,
    SURV_B0,
    TRUE_LOG_HAZARD,
    WEIBULL_SHAPE,
)

# ---------------------------------------------------------------------------
# Competing-risks discharge process (Module 20). See `_generate_competing_risks`.
# Discharge is a genuine competing event on the SAME time axis as death: mean
# time-to-discharge RISES with severity, so the discharge hazard FALLS with
# `sofa_score` (sicker patients leave later, if at all). Own RNG streams so no
# existing course number changes.
# ---------------------------------------------------------------------------
DISCHARGE_BASE_DAYS = 15.0    # mean days to discharge at sofa = 0
DISCHARGE_SOFA_SLOPE = 1.4    # extra mean days per SOFA point (slows discharge)
DISCHARGE_SHAPE = 2.1         # Gamma shape of the discharge-time distribution
# Time-varying vasopressor exposure (Module 20, start-stop format).
VASO_BASE_P = 0.15            # P(ever exposed) intercept
VASO_SOFA_SLOPE = 0.035       # rise in exposure probability per SOFA point


def generate_cohort(rng: np.random.Generator) -> pd.DataFrame:
    """Master table: one row per patient."""
    patient_id = np.arange(1, N + 1)
    age = np.clip(rng.normal(AGE_MEAN, AGE_SD, N), AGE_MIN, AGE_MAX).round().astype(int)
    sex = rng.choice(["weiblich", "maennlich"], N, p=[0.46, 0.54])

    # Body measurements -> BMI
    height = np.clip(rng.normal(172, 9, N), 148, 200).round().astype(int)
    weight = np.clip(rng.normal(82, 18, N), 42, 160).round(1)
    bmi = (weight / (height / 100) ** 2).round(1)

    reason = rng.choice(REASONS, N, p=REASON_P)

    # Comorbidities (age-dependent probability).
    p_diab = np.clip(DIAB_B0 + (age - DIAB_AGE_REF) * DIAB_SLOPE, DIAB_MIN, DIAB_MAX)
    diabetes = (rng.random(N) < p_diab).astype(int)
    p_hyp = np.clip(0.10 + (age - 30) * 0.009, 0.05, 0.8)
    hypertension = (rng.random(N) < p_hyp).astype(int)

    smoking = rng.choice(SMOKING, N, p=SMOKING_P)

    # Admission severity / inflammation. `sofa_score` sits downstream of
    # diabetes and the admission reason -> it is a MEDIATOR on their causal
    # path to death, not a confounder (see data/README.md "true DAG").
    sofa = np.clip(rng.poisson(SOFA_POISSON_MEAN, N) + diabetes
                   + (reason == "Sepsis") * SOFA_SEPSIS_BONUS, SOFA_MIN, SOFA_MAX)
    crp = np.clip(rng.gamma(CRP_SHAPE, CRP_SCALE, N), CRP_MIN, CRP_MAX).round(1)  # mg/l

    # Length of stay (days); descriptive only (NOT the survival time).
    los = np.clip(rng.gamma(2.2, 2.4, N) + sofa * 0.4, 1, 60).round().astype(int)

    # ------------------------------------------------------------------
    # Unified time-to-event process: ONE mechanism drives both the
    # survival curve (fu_zeit_tage/status) and the derived 30-day
    # mortality flag, so classification and survival analyses agree by
    # construction.
    # ------------------------------------------------------------------
    eta = (
        SURV_B0
        + TRUE_LOG_HAZARD["age_centred"] * (age - AGE_MEAN)
        + TRUE_LOG_HAZARD["sofa_score"] * sofa
        + TRUE_LOG_HAZARD["crp_mg_l"] * crp
        + TRUE_LOG_HAZARD["diabetes"] * diabetes
        + TRUE_LOG_HAZARD["sepsis"] * (reason == "Sepsis")
        + TRUE_LOG_HAZARD["active_smoker"] * (smoking == "aktiv")
    )
    # Higher eta -> shorter Weibull scale -> shorter time -> higher hazard.
    scale_i = BASELINE_SCALE * np.exp(-eta / WEIBULL_SHAPE)
    lost_mask = rng.random(N) < LOST_TO_FU_RATE
    c_lost_draw = rng.uniform(CENSOR_LO, CENSOR_HI, N)
    c_lost = np.where(lost_mask, c_lost_draw, np.inf)
    u = rng.random(N)
    t_event = scale_i * (-np.log(u)) ** (1 / WEIBULL_SHAPE)

    fu_zeit_tage = np.clip(
        np.minimum(np.minimum(t_event, c_lost), ADMIN_HORIZON_DAYS), 1, None
    )
    fu_zeit_tage = np.round(fu_zeit_tage).astype(int)
    status = (t_event <= np.minimum(c_lost, ADMIN_HORIZON_DAYS)).astype(int)
    # Derived, not independently drawn: 30-day mortality is exactly the
    # observed event within the 30-day administrative horizon.
    verstorben_30d = ((status == 1) & (fu_zeit_tage <= ADMIN_HORIZON_DAYS)).astype(int)
    # `verstorben_30d` is an OBSERVED-event label: a patient lost to follow-up
    # before day 30 gets 0 although a later death is unobserved. This flag says
    # whether the 30-day outcome is actually known (death observed, or followed
    # to the horizon). Pure derivation, no RNG draw: every other number stays.
    fu_vollstaendig_30d = ((status == 1) | (c_lost >= ADMIN_HORIZON_DAYS)).astype(int)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "alter": age,
        "geschlecht": sex,
        "groesse_cm": height,
        "gewicht_kg": weight,
        "bmi": bmi,
        "aufnahmegrund": reason,
        "diabetes": diabetes,
        "hypertonie": hypertension,
        "raucherstatus": smoking,
        "sofa_score": sofa,
        "crp_mg_l": crp,
        "verweildauer_tage": los,
        "fu_zeit_tage": fu_zeit_tage,
        "status": status,
        "verstorben_30d": verstorben_30d,
        "fu_vollstaendig_30d": fu_vollstaendig_30d,
    })

    # Deliberate "dirt" for module 06 (cleaning): missing BMI and inconsistent
    # sex spelling.
    missing = rng.choice(df.index, size=int(0.06 * N), replace=False)
    df.loc[missing, "bmi"] = np.nan
    df.loc[missing, "gewicht_kg"] = np.nan
    typos = rng.choice(df.index, size=int(0.03 * N), replace=False)
    df.loc[typos, "geschlecht"] = "w"  # inconsistent -> normalise later
    return df


def generate_vitals(rng: np.random.Generator, cohort: pd.DataFrame) -> pd.DataFrame:
    """Long format: several measurements (day 0..3) per patient."""
    rows = []
    for _, p in cohort.iterrows():
        n_days = min(4, int(p["verweildauer_tage"]))
        base_map = 75 - p["sofa_score"] * 1.5
        for day in range(n_days):
            rows.append({
                "patient_id": int(p["patient_id"]),
                "tag": day,
                "herzfrequenz": int(np.clip(rng.normal(88 + p["sofa_score"], 12), 40, 180)),
                "map_mmhg": int(np.clip(rng.normal(base_map, 8), 40, 130)),
                "temperatur_c": round(float(np.clip(rng.normal(37.6, 0.8), 34.5, 41)), 1),
                "spo2_prozent": int(np.clip(rng.normal(95, 3), 70, 100)),
            })
    return pd.DataFrame(rows)


def generate_labs(rng: np.random.Generator, cohort: pd.DataFrame) -> pd.DataFrame:
    """Wide format: selected lab values, with realistic MAR missingness."""
    n = len(cohort)
    df = pd.DataFrame({
        "patient_id": cohort["patient_id"].to_numpy(),
        "leukozyten_g_l": np.clip(rng.normal(11 + cohort["sofa_score"], 4, n), 1, 40).round(1),
        "haemoglobin_g_dl": np.clip(rng.normal(12.5, 2.0, n), 5, 19).round(1),
        "kreatinin_mg_dl": np.clip(rng.gamma(2.0, 0.6, n), 0.3, 9).round(2),
        "laktat_mmol_l": np.clip(rng.gamma(1.6, 1.0, n) + cohort["sofa_score"] * 0.1, 0.4, 18).round(1),
        "natrium_mmol_l": np.clip(rng.normal(139, 4, n), 118, 160).round().astype(int),
    })
    # Lactate is MAR, not MCAR: sicker patients (higher SOFA) get lactate
    # drawn more often clinically -> missingness DECREASES with SOFA. This is
    # a deliberate change from the old uniform-random 18 % dropout (fixes
    # Modules 06 & 12, which teach the MCAR/MAR/MNAR distinction).
    #
    # The driver (`sofa_score`) IS a covariate of the outcome models used in
    # Modules 10/12/18. Consequence, and Module 14 teaches exactly this:
    # complete-case *regression coefficients* stay unbiased, the *marginal*
    # mean of lactate does not.
    sofa = cohort["sofa_score"].to_numpy()
    p_missing = 1 / (1 + np.exp(-(-0.3 - 0.35 * sofa)))
    missing_mask = rng.random(n) < p_missing
    df.loc[missing_mask, "laktat_mmol_l"] = np.nan

    df["bga_ph"] = _generate_bga_ph(cohort)
    return df


def _generate_bga_ph(cohort: pd.DataFrame) -> np.ndarray:
    """Blood-gas pH with OUTCOME-dependent missingness (Module 14).

    Deliberately uses its own RNG stream (SEED + 1). Drawing from the shared
    `rng` would shift every later draw and silently change lactate's
    missingness, the clinical notes, and every figure the course documents.

    labor.csv now carries two missingness mechanisms that teach opposite
    lessons:

      laktat_mmol_l  missing depends on `sofa_score` -- a COVARIATE the
                     analysis model conditions on. Complete-case coefficients
                     are unbiased; the marginal mean is not.

      bga_ph         missing depends on `verstorben_30d` -- the OUTCOME. A
                     follow-up blood gas is far likelier to be absent in
                     patients who deteriorated and died early. Complete-case
                     odds ratios stay roughly unbiased (selecting on Y shifts
                     a logistic intercept, not its slopes), but the intercept
                     and every predicted ABSOLUTE risk are badly biased, and
                     the complete-case cohort's apparent mortality collapses
                     from 15.6 % to ~11.5 %. Multiple imputation with the
                     outcome in the imputation model repairs both.

    Missingness depends only on OBSERVED data, so this is MAR, not MNAR --
    multiple imputation is valid here.
    """
    ph, _ = _bga_ph_truth_and_observed(cohort)
    return ph


# pH generating coefficients live in lib/ground_truth.py (single source of
# truth, shared with the Module-12 Monte-Carlo benchmark). `BGA_DEAD` is
# deliberately strong enough that the association survives 99 missing values:
# at -0.014 the complete-data effect sat at p = 0.046, so EVERY imputation
# method "lost significance" and Module 14 would have taught that multiple
# imputation fails. It does not -- the effect was simply too weak to survive
# the missingness. At -0.024 the true odds ratio is ~2.8 per 0.1 pH drop,
# which is what acidosis actually does in critical illness, and the lesson
# (median imputation destroys a robust finding; MI repairs it) becomes real.


def _bga_ph_truth_and_observed(cohort: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (observed_with_nan, complete_truth). Deterministic given SEED."""
    rng_bga = np.random.default_rng(SEED + 1)
    n = len(cohort)
    sofa = cohort["sofa_score"].to_numpy()
    dead = cohort["verstorben_30d"].to_numpy()

    # Acidosis tracks severity and death: ~OR 2.8 per 0.1 pH drop.
    truth = BGA_INTERCEPT + BGA_SOFA * sofa + BGA_DEAD * dead
    truth = truth + rng_bga.normal(0, BGA_NOISE_SD, n)
    truth = np.clip(truth, 6.90, 7.60).round(3)

    # The missingness draw comes AFTER the value draw from the same stream, so
    # retuning the coefficients above leaves the missingness pattern untouched.
    p_missing = 1 / (1 + np.exp(-(BGA_MISS_B0 + BGA_MISS_DEAD * dead)))
    observed = truth.astype(float).copy()
    observed[rng_bga.random(n) < p_missing] = np.nan
    return observed, truth


def write_bga_truth(cohort: pd.DataFrame) -> Path:
    """Ground truth for `bga_ph` – the values before missingness was applied.

    Only exists because the cohort is synthetic. Module 14 uses it to *show*
    what complete-case analysis and median imputation get wrong, instead of
    merely asserting it. Deliberately a separate file: labor.csv must keep
    looking like a real lab export, which never ships the answers.
    """
    path = HERE / "bga_ph_wahrheit.csv"
    _, truth = _bga_ph_truth_and_observed(cohort)
    pd.DataFrame({
        "patient_id": cohort["patient_id"].to_numpy(),
        "bga_ph_wahr": truth,
    }).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Competing-risks / time-varying outcome (Module 20).
# ---------------------------------------------------------------------------

def _generate_competing_risks(cohort: pd.DataFrame) -> pd.DataFrame:
    """Coherent 3-state competing-risks outcome + a time-varying exposure.

    Deliberately uses its OWN RNG streams (SEED + 2 for the discharge time,
    SEED + 3 for the vasopressor exposure), exactly like `_generate_bga_ph`.
    Drawing from the shared `rng` would shift every later draw and silently
    change lactate missingness, the clinical notes, and every figure the
    course documents -- so nothing already in kohorte.csv/labor.csv moves.

    Generating process (statistically coherent with the documented Weibull
    death process in `generate_cohort`):

      * In-hospital DEATH is the observed cohort event: a patient with
        `status == 1` dies in hospital at `fu_zeit_tage` (<= 30 days) and is
        never discharged. This keeps the true death CIF at exactly the cohort
        30-day mortality (~15.6 %), and the cause-specific Cox hazards match
        Module 17 by construction.

      * DISCHARGE is a genuine competing event drawn on the same day axis for
        every patient who does NOT die in hospital (`status == 0`):

            D_i ~ Gamma(shape = DISCHARGE_SHAPE,
                        mean  = DISCHARGE_BASE_DAYS + DISCHARGE_SOFA_SLOPE * sofa_i)

        The mean RISES with `sofa_score`, so the discharge hazard FALLS with
        severity -- sicker patients leave later, or not before day 30. There
        is NO hard ceiling: D_i can land anywhere up to (and past) day 30.

      * A patient exits at whichever competing event comes first:
            - dies in hospital (state 1, at fu_zeit_tage), else
            - discharged (state 2) if D_i < min(fu_zeit_tage, 30), else
            - administratively censored (state 0) at min(fu_zeit_tage, 30).
        Administrative censoring at day 30 is preserved, and the risk set at
        day 30 stays non-degenerate (~90 patients), unlike the old
        `verweildauer_tage`-based hack that emptied it by day 20.

    Consequence taught in Module 20: 1 - Kaplan-Meier (discharge treated as
    censoring) OVERSHOOTS the true CIF -- visibly (~0.28 vs ~0.16) but
    bounded, the realistic textbook bias rather than an artefact hitting 100 %.

    A time-varying VASOPRESSOR exposure is generated in the same helper so
    that Python and R read byte-identical data and report identical hazard
    ratios (they previously drew from different RNG streams). P(ever exposed)
    and the start day both track baseline `sofa_score` but NOT the outcome, so
    death occurs in both arms and the time-varying Cox model converges without
    the complete separation of a toy example.
    """
    rng_dis = np.random.default_rng(SEED + 2)
    rng_vaso = np.random.default_rng(SEED + 3)

    n = len(cohort)
    sofa = cohort["sofa_score"].to_numpy()
    status = cohort["status"].to_numpy()
    fu = cohort["fu_zeit_tage"].to_numpy().astype(float)

    # Latent discharge time (continuous days); mean rises with SOFA.
    mean_dis = DISCHARGE_BASE_DAYS + DISCHARGE_SOFA_SLOPE * sofa
    d_latent = rng_dis.gamma(DISCHARGE_SHAPE, mean_dis / DISCHARGE_SHAPE)

    event_time = np.empty(n, dtype=int)
    event_state = np.empty(n, dtype=int)
    for i in range(n):
        if status[i] == 1:                       # died in hospital -> state 1
            event_time[i] = int(fu[i])
            event_state[i] = 1
            continue
        end = min(int(fu[i]), ADMIN_HORIZON_DAYS)  # observation end (30 admin, <30 lost)
        if d_latent[i] < end:                    # discharged before obs ends -> state 2
            disc_day = int(round(d_latent[i]))
            disc_day = min(max(disc_day, 1), end - 1)  # keep strictly inside (0, end)
            event_time[i] = disc_day
            event_state[i] = 2
        else:                                    # still in hospital at obs end -> state 0
            event_time[i] = end
            event_state[i] = 0

    # Time-varying vasopressor exposure (own stream); depends on baseline SOFA,
    # never on the outcome. `vasopressor_start_tag` is the day exposure begins,
    # or NaN if the patient is never exposed within their observation window.
    p_treat = np.clip(VASO_BASE_P + VASO_SOFA_SLOPE * sofa, 0.05, 0.85)
    ever_treated = rng_vaso.binomial(1, p_treat)
    mean_wait = np.clip(6 - 0.25 * sofa, 1.0, 6.0)
    start_raw = rng_vaso.exponential(mean_wait)
    start_day = np.floor(start_raw).astype(int) + 1  # first exposed day, >= 1
    exposed = (ever_treated == 1) & (start_day < event_time)
    vaso_start = np.where(exposed, start_day.astype(float), np.nan)

    return pd.DataFrame({
        "patient_id": cohort["patient_id"].to_numpy(),
        "event_time": event_time,
        "event_state": event_state,
        "vasopressor_start_tag": vaso_start,
    })


def write_competing_risks(cohort: pd.DataFrame) -> Path:
    """Coherent competing-risks / time-varying dataset for Module 20.

    A separate file (never columns on kohorte.csv/labor.csv): it carries a
    properly generated discharge competing event and a time-varying exposure
    that neither existed as real columns nor could be inferred honestly from
    the loosely-coupled `verweildauer_tage`. Columns:
        patient_id, event_time, event_state (0=censored@30d, 1=death,
        2=discharge), vasopressor_start_tag (day exposure starts, else empty).
    """
    path = HERE / "konkurrenz_risiken.csv"
    _generate_competing_risks(cohort).to_csv(path, index=False)
    return path


def generate_notes(rng: np.random.Generator, cohort: pd.DataFrame) -> pd.DataFrame:
    """Synthetic German admission notes (free text) for the NLP module.

    Each note contains building blocks correlated with the admission reason and
    course. The label `verschlechterung` (0/1) is WEAKLY encoded in the text
    (phrases like "zunehmende Dyspnoe", "Laktat steigend"). About 22 % of notes
    receive a deliberately wrong-sign or ambiguous phrase to keep the
    classification genuinely non-trivial. Vocabulary is intentionally rich (6-8
    finding phrases per admission reason, 8-10 worsening/stable phrases, varied
    openings and sentence structure) so the text task is not a trivial
    3-sentence-template lookup (fixes Module 31).
    Entirely fictitious.
    """
    opening = [
        "Aufnahme über die Notaufnahme.", "Patient:in zur stationären Aufnahme.",
        "Vorstellung in Begleitung des Rettungsdienstes.", "Elektive Aufnahme nach Zuweisung.",
        "Zuweisung durch den Hausarzt.", "Selbstvorstellung in der Notaufnahme.",
    ]
    findings = {
        "Pneumonie": [
            "produktiver Husten seit drei Tagen", "Fieber bis 39,2 °C",
            "rechtsbasal feuchte Rasselgeräusche", "Infiltrat im Röntgen-Thorax",
            "pleuritischer Thoraxschmerz", "Tachypnoe bei Aufnahme",
            "grippaler Infekt in der Vorgeschichte",
        ],
        "Sepsis": [
            "Verdacht auf fokale Infektion", "Tachykardie und Hypotonie",
            "warme Peripherie", "CRP deutlich erhöht", "Schüttelfrost vor Aufnahme",
            "Procalcitonin erhöht", "Verdacht auf Harnwegsinfekt als Fokus",
        ],
        "Herzinsuffizienz": [
            "Beinödeme beidseits", "Orthopnoe", "gestaute Halsvenen", "NT-proBNP erhöht",
            "Gewichtszunahme innerhalb weniger Tage", "feuchte Rasselgeräusche basal beidseits",
            "bekannte reduzierte Auswurffraktion",
        ],
        "COPD-Exazerbation": [
            "bekannte COPD", "verlängertes Exspirium", "Giemen",
            "Sauerstoffsättigung grenzwertig", "vermehrter Auswurf",
            "Zunahme der Atemnot seit Tagen", "Einsatz der Atemhilfsmuskulatur",
        ],
        "Sonstiges": [
            "unspezifische Beschwerden", "Allgemeinzustand reduziert",
            "weitere Abklärung geplant", "Sturz in der häuslichen Umgebung",
            "Elektrolytstörung bei Aufnahme", "Exsikkose bei verminderter Trinkmenge",
        ],
    }
    # Worsening-signal phrases – strong positive signal but not exclusive.
    worsening_phrases = [
        "Im Verlauf zunehmende Dyspnoe.", "Laktat steigend, Kreislauf instabil.",
        "Sauerstoffbedarf zunehmend.", "Vigilanz im Verlauf gemindert.",
        "Verlegung auf Intensivstation erwogen.", "Respiratorische Verschlechterung im Tagesverlauf.",
        "Steigender Katecholaminbedarf.", "Zunehmende Somnolenz.",
        "Azidose progredient.", "Kreislauf zunehmend katecholaminpflichtig.",
    ]
    # Stable-signal phrases – strong negative signal but not exclusive.
    stable_phrases = [
        "Im Verlauf stabil.", "Gutes Ansprechen auf Therapie.",
        "Kreislauf stabil, kein zusätzlicher O2-Bedarf.", "Mobilisation problemlos.",
        "Kardiopulmonal stabil.", "Sauerstoff reduzierbar.", "Afebril im Verlauf.",
        "Gute Rekonvaleszenz.", "Beschwerdefreiheit im Tagesverlauf.",
    ]
    # Ambiguous phrases – appear in both classes, add noise.
    ambiguous_phrases = [
        "Weitere Beobachtung erforderlich.", "Engmaschige Kontrolle geplant.",
        "Verlauf abwartend.", "Befundkonstellation komplex.", "Rücksprache mit Oberarzt.",
    ]
    # Noise rate: probability that a note gets a wrong-sign or ambiguous ending.
    # ~12 % get wrong-sign, ~10 % get an ambiguous phrase (~22 % combined).
    WRONG_SIGN_RATE = 0.18
    AMBIGUOUS_RATE  = 0.04

    rows = []
    for _, p in cohort.iterrows():
        # Worsening correlates with SOFA + mortality (stochastic, not deterministic).
        p_worse = 1 / (1 + np.exp(-(-1.6 + 0.18 * p["sofa_score"] + 1.4 * p["verstorben_30d"])))
        worse = int(rng.random() < p_worse)

        # Draw the ending phrase with deliberate noise to prevent perfect separation.
        draw = rng.random()
        if draw < AMBIGUOUS_RATE:
            course = rng.choice(ambiguous_phrases)
        elif draw < AMBIGUOUS_RATE + WRONG_SIGN_RATE:
            # Wrong-sign: positive case gets a stable phrase, and vice versa.
            course = rng.choice(stable_phrases if worse else worsening_phrases)
        else:
            course = rng.choice(worsening_phrases if worse else stable_phrases)

        n_findings = int(rng.integers(2, 4))  # 2 or 3 findings per note
        blocks = list(rng.choice(findings[p["aufnahmegrund"]], size=n_findings, replace=False))
        findings_text = ", ".join(blocks)

        # Vary sentence structure: sometimes fold the demographic clause into
        # the findings sentence (2 sentences total), sometimes keep it
        # separate (3 sentences total).
        if rng.integers(0, 2) == 0:
            text = (f"{rng.choice(opening)} {p['alter']}-jährige:r Patient:in, "
                     f"{findings_text}. {course}")
        else:
            text = (f"{rng.choice(opening)} {p['alter']}-jährige:r Patient:in. "
                     f"{findings_text[0].upper()}{findings_text[1:]}. {course}")

        rows.append({"patient_id": int(p["patient_id"]), "notiz": text, "verschlechterung": worse})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module-04 companion files (data-import formats).
# ---------------------------------------------------------------------------

def write_german_csv(cohort: pd.DataFrame) -> Path:
    """German hospital export format: sep=';', latin-1, decimal comma.

    A real KIS export spells umlauts out ("männlich", not "maennlich"). Without
    them the file is pure ASCII, reads fine as UTF-8 too, and Modul 04's
    encoding lesson has nothing to show.
    """
    path = HERE / "kohorte_de.csv"
    export = cohort.assign(geschlecht=cohort["geschlecht"].replace({"maennlich": "männlich"}))
    path.write_text(
        export.to_csv(sep=";", decimal=",", index=False),
        encoding="latin-1",
    )
    return path


def write_inclusion_workbook(cohort: pd.DataFrame, labs: pd.DataFrame) -> Path:
    """3-sheet inclusion-list workbook: Kohorte, Labor, Klinik."""
    path = HERE / "einschluss.xlsx"
    n_rows = 40
    subset_ids = cohort["patient_id"].head(n_rows)

    sheet_kohorte = cohort.loc[cohort["patient_id"].isin(subset_ids),
                                ["patient_id", "alter", "aufnahmegrund", "sofa_score"]]
    # Columns A:F on the Labor sheet must be meaningful (Modul 04 reads
    # usecols="A:F") -> exactly 6 columns.
    sheet_labor = labs.loc[labs["patient_id"].isin(subset_ids),
                            ["patient_id", "leukozyten_g_l", "haemoglobin_g_dl",
                             "kreatinin_mg_dl", "laktat_mmol_l", "natrium_mmol_l"]]
    sheet_klinik = cohort.loc[cohort["patient_id"].isin(subset_ids),
                               ["patient_id", "crp_mg_l", "verweildauer_tage", "verstorben_30d"]]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet_kohorte.to_excel(writer, sheet_name="Kohorte", index=False)
        sheet_labor.to_excel(writer, sheet_name="Labor", index=False)
        sheet_klinik.to_excel(writer, sheet_name="Klinik", index=False)
        props = writer.book.properties
        props.creator = props.lastModifiedBy = "data/generate_data.py"
        props.created = props.modified = XLSX_EPOCH
    _freeze_workbook(path)
    return path


def _freeze_workbook(path: Path) -> None:
    """Make the .xlsx reproducible.

    openpyxl stamps the current wall-clock time into every zip member header and
    overwrites <dcterms:modified> during save, so two runs of this script produce
    different bytes for identical data. Rewrite the archive with one fixed
    timestamp. Everything else – member order, compression, permissions – is
    preserved, and the workbook stays readable by pandas, Excel and readxl.
    """
    stamp = XLSX_EPOCH.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    modified = re.compile(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")

    with zipfile.ZipFile(path) as src:
        members = [(info, src.read(info.filename)) for info in src.infolist()]

    tmp = path.with_suffix(".xlsx.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info, payload in members:
            if info.filename == "docProps/core.xml":
                payload = modified.sub(rb"\g<1>" + stamp + rb"\g<2>", payload)
            frozen = zipfile.ZipInfo(info.filename, date_time=XLSX_EPOCH.timetuple()[:6])
            frozen.compress_type = info.compress_type
            frozen.external_attr = info.external_attr
            dst.writestr(frozen, payload)
    tmp.replace(path)


def write_flat_json(cohort: pd.DataFrame) -> Path:
    """Small flat JSON array (REDCap-like export) -> loads directly as a frame."""
    path = HERE / "beispiel.json"
    records = cohort[["patient_id", "alter", "aufnahmegrund"]].head(10).to_dict(orient="records")
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_fhir_bundle(cohort: pd.DataFrame) -> Path:
    """Minimal, fictitious FHIR R4 Bundle with Patient + Observation entries."""
    path = HERE / "fhir_bundle.json"
    sample = cohort[["patient_id", "alter", "geschlecht", "sofa_score", "crp_mg_l"]].head(3)

    entries = []
    for _, r in sample.iterrows():
        pid = f"pat-{int(r['patient_id'])}"
        gender = "female" if r["geschlecht"] in ("weiblich", "w") else "male"
        entries.append({
            "resource": {
                "resourceType": "Patient",
                "id": pid,
                "gender": gender,
                "birthDate": f"{2024 - int(r['alter'])}-01-01",
            }
        })
        entries.append({
            "resource": {
                "resourceType": "Observation",
                "id": f"obs-sofa-{int(r['patient_id'])}",
                "status": "final",
                "code": {"text": "SOFA score"},
                "subject": {"reference": f"Patient/{pid}"},
                "valueQuantity": {"value": int(r["sofa_score"]), "unit": "score"},
            }
        })
        entries.append({
            "resource": {
                "resourceType": "Observation",
                "id": f"obs-crp-{int(r['patient_id'])}",
                "status": "final",
                "code": {"text": "C-reactive protein"},
                "subject": {"reference": f"Patient/{pid}"},
                "valueQuantity": {"value": float(r["crp_mg_l"]), "unit": "mg/L"},
            }
        })

    bundle = {"resourceType": "Bundle", "type": "collection", "entry": entries}
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Ground-truth verification (printed at the end of main()).
# ---------------------------------------------------------------------------

def verify_survival(cohort: pd.DataFrame) -> dict:
    from lifelines import CoxPHFitter, KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(cohort["fu_zeit_tage"], event_observed=cohort["status"])
    surv = kmf.survival_function_["KM_estimate"].to_numpy()
    monotone = bool(np.all(np.diff(surv) <= 1e-12))

    cox_df = cohort.copy()
    cox_df["sepsis"] = (cox_df["aufnahmegrund"] == "Sepsis").astype(int)
    cox_df["aktiv"] = (cox_df["raucherstatus"] == "aktiv").astype(int)
    cph = CoxPHFitter()
    cph.fit(
        cox_df[["fu_zeit_tage", "status", "alter", "sofa_score", "crp_mg_l",
                "diabetes", "sepsis", "aktiv"]],
        duration_col="fu_zeit_tage", event_col="status",
    )
    hr = np.exp(cph.params_)
    return {
        "monotone": monotone,
        "n_censored": int((cohort["status"] == 0).sum()),
        "hr_sepsis": float(hr["sepsis"]),
        "hr_alter": float(hr["alter"]),
        "hr_sofa": float(hr["sofa_score"]),
        "cph": cph,
    }


def verify_lactate_mar(cohort: pd.DataFrame, labs: pd.DataFrame) -> dict:
    import statsmodels.api as sm

    merged = cohort[["patient_id", "alter", "sofa_score"]].merge(labs, on="patient_id")
    is_missing = merged["laktat_mmol_l"].isna().astype(int)

    X = sm.add_constant(merged[["sofa_score"]])
    model = sm.Logit(is_missing, X).fit(disp=0)
    coef_sofa, p_sofa = model.params["sofa_score"], model.pvalues["sofa_score"]

    X_age = sm.add_constant(merged[["alter"]])
    model_age = sm.Logit(is_missing, X_age).fit(disp=0)
    p_age = model_age.pvalues["alter"]

    return {
        "missing_frac": float(is_missing.mean()),
        "coef_sofa": float(coef_sofa),
        "p_sofa": float(p_sofa),
        "p_age": float(p_age),
    }


def verify_bga_ph(cohort: pd.DataFrame, labs: pd.DataFrame) -> dict:
    """Checks the Module-12 teaching claims hold for the generated data."""
    import statsmodels.formula.api as smf

    m = cohort[["patient_id", "alter", "sofa_score", "verstorben_30d"]].merge(labs, on="patient_id")
    missing = m["bga_ph"].isna()
    complete = m.dropna(subset=["bga_ph"])

    # Complete-case logistic fit vs. the apparent mortality of complete cases.
    cc = smf.logit("verstorben_30d ~ bga_ph + alter + sofa_score", complete).fit(disp=0)
    return {
        "missing_frac": float(missing.mean()),
        "missing_survivors": float(missing[m["verstorben_30d"] == 0].mean()),
        "missing_deceased": float(missing[m["verstorben_30d"] == 1].mean()),
        "mortality_all": float(m["verstorben_30d"].mean()),
        "mortality_complete_case": float(complete["verstorben_30d"].mean()),
        "cc_or_per_01_ph_drop": float(np.exp(-0.1 * cc.params["bga_ph"])),
    }


def verify_competing_risks(cohort: pd.DataFrame, cr: pd.DataFrame) -> dict:
    """Checks the Module-30 teaching claims hold for the generated data.

    Confirms: (a) the true death CIF at day 30 matches cohort mortality, (b)
    naive 1-KM (discharge as censoring) overshoots it visibly but stays
    bounded, (c) the day-30 risk set is non-degenerate, and (d) the
    time-varying vasopressor Cox model converges with a plausible HR.
    """
    import warnings

    from lifelines import AalenJohansenFitter, CoxTimeVaryingFitter, KaplanMeierFitter

    et = cr["event_time"].to_numpy(dtype=float)
    es = cr["event_state"].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Seeded: lifelines jitters tied event times; a fixed seed keeps the
        # CIF reproducible. Query at t <= 30 (the last jittered row can sit at
        # 30+epsilon with a tiny risk set and misreport the value).
        ajf = AalenJohansenFitter(seed=SEED)
        ajf.fit(et, es, event_of_interest=1)
    cif_death_30 = float(ajf.cumulative_density_.loc[:ADMIN_HORIZON_DAYS].iloc[-1, 0])

    kmf = KaplanMeierFitter()
    kmf.fit(et, event_observed=(es == 1))
    naive_km_30 = float(1 - kmf.survival_function_.iloc[-1, 0])

    # Risk set at (just before) day 30: patients whose event/censoring is >= 30.
    risk_set_30 = int((et >= ADMIN_HORIZON_DAYS).sum())

    # Fine-Gray > cause-specific SOFA check needs R; here we sanity-check the
    # driver: discharge (state 2) hazard falls with SOFA, so higher-SOFA
    # patients are LESS likely to be discharged.
    merged = cohort[["patient_id", "sofa_score"]].merge(cr, on="patient_id")
    hi = merged["sofa_score"] >= merged["sofa_score"].median()
    disch_rate_hi = float((merged.loc[hi, "event_state"] == 2).mean())
    disch_rate_lo = float((merged.loc[~hi, "event_state"] == 2).mean())

    # Time-varying vasopressor Cox convergence + HR (built exactly like the
    # module scripts do from the shared data).
    sofa = merged.set_index("patient_id")["sofa_score"]
    rows = []
    for _, r in cr.iterrows():
        pid = int(r["patient_id"])
        t_end = float(r["event_time"])
        death = int(r["event_state"] == 1)
        start = r["vasopressor_start_tag"]
        s = float(sofa[pid])
        if pd.notna(start) and start < t_end:
            rows.append(dict(patient_id=pid, start=0.0, stop=float(start),
                             vasopressor=0, death=0, sofa=s))
            rows.append(dict(patient_id=pid, start=float(start), stop=t_end,
                             vasopressor=1, death=death, sofa=s))
        else:
            rows.append(dict(patient_id=pid, start=0.0, stop=t_end,
                             vasopressor=0, death=death, sofa=s))
    tv = pd.DataFrame(rows)
    tv = tv[tv["stop"] > tv["start"]]
    ctv = CoxTimeVaryingFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctv.fit(tv, id_col="patient_id", start_col="start", stop_col="stop",
                event_col="death", show_progress=False)
    hr_vaso = float(np.exp(ctv.params_["vasopressor"]))
    hr_sofa_tv = float(np.exp(ctv.params_["sofa"]))

    return {
        "n_death": int((es == 1).sum()),
        "n_discharge": int((es == 2).sum()),
        "n_censored": int((es == 0).sum()),
        "cif_death_30": cif_death_30,
        "naive_km_30": naive_km_30,
        "risk_set_30": risk_set_30,
        "disch_rate_hi_sofa": disch_rate_hi,
        "disch_rate_lo_sofa": disch_rate_lo,
        "n_vaso_exposed": int(cr["vasopressor_start_tag"].notna().sum()),
        "hr_vaso": hr_vaso,
        "hr_sofa_tv": hr_sofa_tv,
    }


def verify_notes(notes: pd.DataFrame) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(notes["notiz"])
    y = notes["verschlechterung"].to_numpy()

    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

    clf.fit(X, y)
    top_idx = np.argsort(np.abs(clf.coef_[0]))[::-1][:15]
    feature_names = np.array(vec.get_feature_names_out())
    top_tokens = list(feature_names[top_idx])

    return {
        "positive_rate": float(y.mean()),
        "cv_auc_mean": float(aucs.mean()),
        "cv_auc_std": float(aucs.std()),
        "top_tokens": top_tokens,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    cohort = generate_cohort(rng)
    vitals = generate_vitals(rng, cohort)
    labs = generate_labs(rng, cohort)
    notes = generate_notes(rng, cohort)

    cohort.to_csv(HERE / "kohorte.csv", index=False)
    vitals.to_csv(HERE / "vitalwerte.csv", index=False)
    labs.to_csv(HERE / "labor.csv", index=False)
    notes.to_csv(HERE / "notizen.csv", index=False)

    de_path = write_german_csv(cohort)
    xlsx_path = write_inclusion_workbook(cohort, labs)
    json_path = write_flat_json(cohort)
    fhir_path = write_fhir_bundle(cohort)
    write_bga_truth(cohort)
    cr_path = write_competing_risks(cohort)
    cr = pd.read_csv(cr_path)

    # Remove the superseded stray demo file, if present.
    stray = HERE / "kohorte_de_demo.csv"
    if stray.exists():
        stray.unlink()

    print(f"Generated (seed {SEED}):")
    print(f"  kohorte.csv     {cohort.shape[0]:>5} rows, {cohort.shape[1]} cols")
    print(f"  vitalwerte.csv  {vitals.shape[0]:>5} rows, {vitals.shape[1]} cols")
    print(f"  labor.csv       {labs.shape[0]:>5} rows, {labs.shape[1]} cols")
    print(f"  notizen.csv     {notes.shape[0]:>5} rows, {notes.shape[1]} cols")
    print(f"  {de_path.name}")
    print(f"  {xlsx_path.name}")
    print(f"  {json_path.name}")
    print(f"  {fhir_path.name}")
    print(f"  {cr_path.name:15} {cr.shape[0]:>5} rows, {cr.shape[1]} cols")
    # Tenth file: the Module-12 ground truth. It was written but never reported,
    # so the printed summary contradicted data/README.md's "zehn Dateien".
    print(f"  bga_ph_wahrheit.csv   {N:>3} rows, 2 cols")

    # --------------------------- verification ---------------------------
    surv = verify_survival(cohort)
    mar = verify_lactate_mar(cohort, labs)
    bga = verify_bga_ph(cohort, labs)
    cr_v = verify_competing_risks(cohort, cr)
    txt = verify_notes(notes)

    n_events = int(cohort["status"].sum())
    mortality = cohort["verstorben_30d"].mean()
    median_fu = float(cohort["fu_zeit_tage"].median())
    censoring_pct = surv["n_censored"] / N

    print("\n=== Verification ===")
    print(f"KM curve monotone non-increasing: {surv['monotone']}  "
          f"(censored: {surv['n_censored']}/{N} = {censoring_pct:.1%})")
    print(f"Cox HRs -> sepsis: {surv['hr_sepsis']:.2f}  age: {surv['hr_alter']:.3f}  "
          f"sofa: {surv['hr_sofa']:.2f}  (all should be > 1)")
    print(f"Lactate missing: {mar['missing_frac']:.1%}  "
          f"SOFA coef (is_missing~sofa): {mar['coef_sofa']:.3f}  p={mar['p_sofa']:.4f}  "
          f"(age p={mar['p_age']:.3f}, should be non-significant)")
    print(f"BGA pH missing: {bga['missing_frac']:.1%}  "
          f"(survivors {bga['missing_survivors']:.1%} vs deceased {bga['missing_deceased']:.1%} "
          f"-> missingness depends on the OUTCOME)")
    print(f"  apparent mortality among complete cases: {bga['mortality_complete_case']:.1%} "
          f"(true: {bga['mortality_all']:.1%})  <- complete-case selection bias")
    print(f"Notes positive rate (verschlechterung): {txt['positive_rate']:.1%}  "
          f"CV-AUC: {txt['cv_auc_mean']:.3f} +/- {txt['cv_auc_std']:.3f}")
    print(f"Top-15 tokens: {txt['top_tokens']}")
    print(f"Competing risks -> states death/discharge/censored: "
          f"{cr_v['n_death']}/{cr_v['n_discharge']}/{cr_v['n_censored']}")
    print(f"  true death CIF@30d: {cr_v['cif_death_30']:.1%}  "
          f"naive 1-KM@30d: {cr_v['naive_km_30']:.1%}  "
          f"(overshoots but bounded)  risk set@30d: {cr_v['risk_set_30']}")
    print(f"  discharge rate hi-SOFA {cr_v['disch_rate_hi_sofa']:.1%} < lo-SOFA "
          f"{cr_v['disch_rate_lo_sofa']:.1%}  -> higher SOFA lowers the discharge hazard")
    print(f"  vasopressor exposed: {cr_v['n_vaso_exposed']}  "
          f"time-varying Cox HR vaso: {cr_v['hr_vaso']:.2f}  sofa: {cr_v['hr_sofa_tv']:.2f}")

    print("\n" + "=" * 70)
    print("GROUND TRUTH SUMMARY")
    print("=" * 70)
    print(f"N = {N}")
    print(f"30-day mortality (verstorben_30d): {mortality:.1%}  ({n_events} events)")
    print(f"Median fu_zeit_tage: {median_fu:.0f} days")
    print(f"Censoring: {censoring_pct:.1%}")
    print(f"Cox HRs -> sepsis: {surv['hr_sepsis']:.2f}, age: {surv['hr_alter']:.3f}, "
          f"sofa: {surv['hr_sofa']:.2f}")
    print(f"Lactate missing: {mar['missing_frac']:.1%}, SOFA coef: {mar['coef_sofa']:.3f} "
          f"(p={mar['p_sofa']:.4f})")
    print(f"Notes positive rate: {txt['positive_rate']:.1%}, CV-AUC: {txt['cv_auc_mean']:.3f}")
    print(f"Top tokens (sample): {txt['top_tokens'][:8]}")
    # Enumerate what is actually on disk rather than a hand-maintained list:
    # the previous literal omitted bga_ph_wahrheit.csv and claimed nine files
    # where ten were written.
    geschrieben = sorted(
        p.name for p in HERE.iterdir()
        if p.suffix.lower() in {".csv", ".json", ".xlsx"}
    )
    print(f"Files written ({len(geschrieben)}): " + ", ".join(geschrieben))


if __name__ == "__main__":
    main()
