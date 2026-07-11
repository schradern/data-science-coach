# Module 34 - why silent deployment catches what a one-off retrospective
# validation misses: simulate the shared cohort's model performance under a
# gradual population shift (a sicker ICU population over time), which is
# exactly the kind of drift a real silent-deployment phase monitors for.
#   Rscript module/34-design-ki-studien/code/r.R

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

set.seed(SEED)

checklist <- data.frame(
  phase = c(
    "Retrospective validation",
    "Silent deployment",
    "Clinical trial",
    "Post-market monitoring"
  ),
  primary_evidence = c(
    "AUC, calibration, subgroups",
    "Prospective performance, latency, missing inputs",
    "Patient outcome, process metric, safety endpoint",
    "Drift, alert burden, override rate"
  ),
  main_risk = c(
    "Historical bias",
    "No patient impact yet",
    "Intervention risk",
    "Model decay"
  )
)

cat("=== Clinical AI validation ladder ===\n")
print(checklist, row.names = FALSE)
cat("\nDecision rule: do not move to patient-facing use before silent deployment\n")
cat("shows stable data flow, calibration, and subgroup performance.\n")

# Rank-based AUC (Mann-Whitney U) and Brier score -- no extra package needed.
auc_fn <- function(y, p) {
  r <- rank(p)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
brier_fn <- function(y, p) mean((p - y)^2)

df <- load_cohort()
n <- nrow(df)
train_idx <- sample(seq_len(n), size = round(0.7 * n))
train <- df[train_idx, ]
test <- df[-train_idx, ]

fit <- glm(verstorben_30d ~ alter + sofa_score + crp_mg_l + diabetes,
           data = train, family = binomial)

p_test <- predict(fit, newdata = test, type = "response")
auc0 <- auc_fn(test$verstorben_30d, p_test)
brier0 <- brier_fn(test$verstorben_30d, p_test)

cat("\n=== Simuliertes Silent Deployment: Population wird schrittweise kraenker ===\n")
cat("(dasselbe, eingefrorene Modell -- nur die Population verschiebt sich)\n")
cat(sprintf("Monat 0: mean_sofa=%.2f  AUC=%.3f  Brier=%.3f\n", mean(test$sofa_score), auc0, brier0))

# Every drift month must be scored OUT OF SAMPLE, exactly like month 0.
# Resample only from the held-out TEST set (never the full cohort, which is
# ~70% training patients the frozen model already saw); otherwise the
# drift-month metrics would be largely in-sample and optimistically biased.
sofa_mean <- mean(test$sofa_score); sofa_sd <- sd(test$sofa_score)
n_test <- nrow(test)
n_resample <- 3000
n_repeats <- 25
n_months <- 6
auc_final <- auc0
brier_final <- brier0
for (m in 1:n_months) {
  shift <- 0.15 * m
  w <- exp(shift * (test$sofa_score - sofa_mean) / sofa_sd)
  pr <- w / sum(w)
  aucs <- numeric(n_repeats); briers <- numeric(n_repeats); sofas <- numeric(n_repeats)
  for (r in 1:n_repeats) {
    samp <- test[sample(seq_len(n_test), size = n_resample, replace = TRUE, prob = pr), ]
    p_samp <- predict(fit, newdata = samp, type = "response")
    aucs[r] <- auc_fn(samp$verstorben_30d, p_samp)
    briers[r] <- brier_fn(samp$verstorben_30d, p_samp)
    sofas[r] <- mean(samp$sofa_score)
  }
  cat(sprintf("Monat %d: mean_sofa=%.2f  AUC=%.3f  Brier=%.3f\n",
              m, mean(sofas), mean(aucs), mean(briers)))
  if (m == n_months) {
    auc_final <- mean(aucs)
    brier_final <- mean(briers)
  }
}

auc_change <- auc_final - auc0
brier_change <- brier_final - brier0
cat(sprintf("\nAUC von Monat 0 zu Monat %d: %+.3f\n", n_months, auc_change))
cat(sprintf("Brier Score von Monat 0 zu Monat %d: %+.3f (%+.0f%% relativ)\n",
            n_months, brier_change, 100 * brier_change / brier0))
if (brier_change > 0.02 && auc_change > -0.02) {
  cat("-> Die Diskriminierung (AUC) bleibt stabil oder verbessert sich sogar, waehrend sich die\n")
  cat("   Kalibrierung (Brier Score) deutlich verschlechtert: die vorhergesagten Risiken passen\n")
  cat("   nicht mehr zur tatsaechlichen Ereignisrate der (nun kraenkeren) Population. Eine einmalige\n")
  cat("   retrospektive AUC-Zahl haette das nie gezeigt -- genau dafuer beobachtet Silent Deployment\n")
  cat("   die Kalibrierung fortlaufend mit.\n")
} else if (auc_change < -0.02) {
  cat("-> Sowohl Diskriminierung als auch Kalibrierung verschlechtern sich unter der\n")
  cat("   verschobenen Population.\n")
} else {
  cat("-> In diesem Lauf zeigt sich kein eindeutiger Drift-Effekt -- Ergebnis ist zufallsabhaengig\n")
  cat("   (SEED, Resampling-Staerke); wiederhole mit anderem SEED, um Robustheit zu pruefen.\n")
}
