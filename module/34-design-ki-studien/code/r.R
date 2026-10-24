# Module 34 - why silent deployment catches what a one-off retrospective
# validation misses: freeze a model, then let the true risk drift while the
# patients' recorded features look the same. The drift is a genuine change of
# P(Y|X): month by month the log-odds of death rise by a fixed offset for every
# patient profile (factors the model does not see). It is simulated
# deterministically by up-weighting the observed deaths of the held-out
# patients -- weight exp(delta) per event multiplies the event odds at each X by
# exp(delta). No random resampling, so R and code/python.py print the same numbers.
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

# Rank-based AUC (Mann-Whitney U) -- no extra package needed. With weights that
# are constant within each class the weighted AUC equals this unweighted one:
# a pure baseline-risk shift leaves the ranking untouched.
auc_fn <- function(y, p) {
  r <- rank(p)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

features <- c("alter", "sofa_score", "crp_mg_l", "diabetes")
validation_every <- 3           # every third patient (patient_id divisible by 3) is held out
n_months <- 6
logodds_shift_per_month <- 0.15 # true log-odds of death rise by 0.15 per month at every X

df <- load_cohort()
is_test <- df$patient_id %% validation_every == 0
dev <- df[!is_test, ]
test <- df[is_test, ]

fit <- glm(verstorben_30d ~ alter + sofa_score + crp_mg_l + diabetes,
           data = dev, family = binomial)
p <- predict(fit, newdata = test, type = "response")
lp <- qlogis(p)
y <- test$verstorben_30d

cat("\n=== Simuliertes Silent Deployment: Grundrisiko steigt, Modell bleibt eingefroren ===\n")
cat(sprintf("(entwickelt an %d Patient:innen, bewertet an %d zurueckgehaltenen; Log-Odds +%.2f pro Monat)\n",
            nrow(dev), nrow(test), logodds_shift_per_month))

rows <- lapply(0:n_months, function(month) {
  delta <- logodds_shift_per_month * month
  w <- ifelse(y == 1, exp(delta), 1)
  observed <- sum(w * y) / sum(w)
  predicted <- sum(w * p) / sum(w)
  # quasibinomial: same point estimates as binomial, no warning for non-integer weights
  slope <- unname(coef(glm(y ~ lp, weights = w, family = quasibinomial))[2])
  data.frame(month = month, observed = observed, predicted = predicted,
             oe = observed / predicted, slope = slope, auc = auc_fn(y, p),
             brier = sum(w * (p - y)^2) / sum(w))
})
drift <- do.call(rbind, rows)
print(round(drift, 3), row.names = FALSE)

first <- drift[1, ]; last <- drift[nrow(drift), ]
cat(sprintf("\nAUC von Monat 0 zu Monat %d: %+.3f\n", n_months, round(last$auc - first$auc, 3) + 0))
cat(sprintf("Kalibrierungs-Steigung von Monat 0 zu Monat %d: %+.3f\n", n_months,
            round(last$slope - first$slope, 3) + 0))
cat(sprintf("O:E von Monat 0 zu Monat %d: %.2f -> %.2f\n", n_months, first$oe, last$oe))
cat(sprintf("Brier Score von Monat 0 zu Monat %d: %+.3f (bewegt sich auch mit der Ereignisrate, kein reines Kalibrierungsmass)\n",
            n_months, last$brier - first$brier))
cat("-> Rangfolge (AUC) und Kalibrierungs-Steigung bleiben unveraendert, waehrend die\n")
cat("   vorhergesagten Risiken immer weiter unter der beobachteten Sterblichkeit liegen\n")
cat("   (O:E steigt). Eine einmalige retrospektive AUC-Zahl haette das nie gezeigt --\n")
cat("   genau dafuer beobachtet Silent Deployment die Kalibrierung fortlaufend mit.\n")
