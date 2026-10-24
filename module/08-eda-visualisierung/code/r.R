# Module 08 – Exploratory data analysis and visualisation (R / ggplot2)
#
# Runs standalone from the project root:
#   Rscript module/08-eda-visualisierung/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
})

# Resolve project root relative to this script.
.script      <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
.root        <- dirname(dirname(dirname(dirname(.script))))
source(file.path(.root, "lib", "helpers.R"))

# Figures belong in the module's assets/ dir (as in every other module), not in
# code/. R writes "_r"-suffixed variants of the same three chapter figures; the
# canonical no-suffix versions the README embeds come from data/figures.py.
.figure_dir <- file.path(dirname(dirname(.script)), "assets")

# ── Prepare data ──────────────────────────────────────────────────────────────

cohort <- load_cohort() |>
  mutate(geschlecht = if_else(geschlecht == "w", "weiblich", geschlecht))
df <- left_join(cohort, load_labs(), by = "patient_id")

# ── 1) Numeric overview ───────────────────────────────────────────────────────

cat("=== 1) Overview: summary() ===\n")
numeric_cols <- c("alter", "bmi", "sofa_score", "crp_mg_l",
                  "verweildauer_tage", "laktat_mmol_l", "kreatinin_mg_dl")
print(summary(df[numeric_cols]))

cat("\n=== 2) Frequencies: count() ===\n")
for (col in c("aufnahmegrund", "geschlecht", "raucherstatus")) {
  cat(sprintf("\n--- %s ---\n", col))
  df |> count(.data[[col]], sort = TRUE) |> print()
}

cat("\n=== 3) Missing values per column ===\n")
missing <- colSums(is.na(df))
print(missing[missing > 0])

# ── 2) Outlier detection (IQR method) ────────────────────────────────────────

cat("\n=== 4) Outlier detection (IQR method) ===\n")
for (col in c("crp_mg_l", "laktat_mmol_l")) {
  vals <- df[[col]]
  q1  <- quantile(vals, 0.25, na.rm = TRUE)
  q3  <- quantile(vals, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  n_out <- sum(vals < (q1 - 1.5 * iqr) | vals > (q3 + 1.5 * iqr), na.rm = TRUE)
  cat(sprintf("%s: IQR=%.2f, bounds=[%.2f, %.2f] -> %d flagged  (verify clinically)\n",
              col, iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr, n_out))
}

# ── 3) Correlation matrix ─────────────────────────────────────────────────────

cat("\n=== 5) Correlation matrix (Pearson) ===\n")
corr_cols <- c("alter", "sofa_score", "crp_mg_l", "laktat_mmol_l",
               "verweildauer_tage", "verstorben_30d")
print(round(cor(df[corr_cols], use = "pairwise.complete.obs"), 2))

# ── 4) Figures ────────────────────────────────────────────────────────────────

cat("\n=== Generating figures ===\n")

# Histogram: age distribution by mortality
p_hist <- ggplot(df, aes(x = alter,
                          fill = factor(verstorben_30d, labels = c("Überlebt", "Verstorben")))) +
  geom_histogram(bins = 20, alpha = 0.65, position = "identity", colour = "white") +
  scale_fill_manual(values = c("Überlebt" = "#2A5C8A", "Verstorben" = "#B5482E")) +
  labs(x = "Alter (Jahre)", y = "Anzahl Patient:innen",
       title = "Altersverteilung nach 30-Tage-Mortalität",
       fill = NULL) +
  theme_minimal()

path_hist <- file.path(.figure_dir, "verteilung_alter_r.png")
ggsave(path_hist, p_hist, width = 7, height = 4, dpi = 120)
cat("Saved:", path_hist, "\n")

# Boxplot: lactate by admission type, sorted by median
order_lvl <- df |>
  group_by(aufnahmegrund) |>
  summarise(m = median(laktat_mmol_l, na.rm = TRUE)) |>
  arrange(desc(m)) |>
  pull(aufnahmegrund)

p_box <- ggplot(df |> filter(!is.na(laktat_mmol_l)),
                aes(x = factor(aufnahmegrund, levels = order_lvl),
                    y = laktat_mmol_l, fill = aufnahmegrund)) +
  geom_boxplot(outlier.size = 1.5, outlier.alpha = 0.5, show.legend = FALSE) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = "Aufnahmegrund", y = "Laktat (mmol/l)",
       title = "Laktatverteilung nach Aufnahmegrund") +
  theme_minimal()

path_box <- file.path(.figure_dir, "verteilung_laktat_nach_grund_r.png")
ggsave(path_box, p_box, width = 8, height = 4, dpi = 120)
cat("Saved:", path_box, "\n")

# Scatter: CRP vs. length of stay, coloured by mortality, with linear trend line
# – the same figure the chapter (§7) discusses.
p_scatter <- ggplot(df,
                    aes(x = crp_mg_l, y = verweildauer_tage)) +
  geom_point(aes(colour = factor(verstorben_30d, labels = c("Überlebt", "Verstorben"))),
             alpha = 0.42, size = 1.5) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = "#555555", linetype = "dashed", linewidth = 0.6) +
  scale_colour_manual(values = c("Überlebt" = "#2A5C8A", "Verstorben" = "#B5482E")) +
  labs(x = "CRP (mg/l)", y = "Verweildauer (Tage)",
       title = "CRP vs. Verweildauer nach 30-Tage-Mortalität",
       colour = NULL) +
  theme_minimal()

path_scatter <- file.path(.figure_dir, "streu_crp_verweildauer_r.png")
ggsave(path_scatter, p_scatter, width = 7, height = 5, dpi = 120)
cat("Saved:", path_scatter, "\n")

cat("\nDone.\n")
