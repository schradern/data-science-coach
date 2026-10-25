# Module 32 – Deployment, monitoring, fairness and governance (R).
#   Rscript module/32-einsatz-governance/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; dataset column names stay German.
# Same steps as code/python.py: text classifier (TF-IDF + penalised logistic
# regression), model persistence, drift simulation, subgroup AUC with
# bootstrap 95%-CI.

suppressPackageStartupMessages({
  missing_pkgs <- setdiff(c("tidyverse", "tidytext", "glmnet", "yardstick"),
                          rownames(installed.packages()))
  if (length(missing_pkgs) > 0) {
    message("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
    message("Install with: install.packages(c('tidyverse', 'tidytext', 'glmnet', 'yardstick'))")
    quit(save = "no", status = 1)
  }
  library(tidyverse)
  library(tidytext)
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

# Drift simulation: flip this fraction of test labels per monitoring step.
DRIFT_FLIP_RATES <- c(0.00, 0.05, 0.10, 0.15, 0.20)
N_BOOTSTRAP <- 1000

auc <- function(y, proba) {
  yardstick::roc_auc_vec(factor(y, levels = c(1, 0)), proba, event_level = "first")
}

# ---- 1) Train text classifier -----------------------------------------------
# Split first, then vocabulary and IDF on the training notes only (as in
# Module 31): unigrams + bigrams, min_df = 2, max_df = 0.95, sublinear tf.
cohort <- load_cohort() |>
  mutate(geschlecht_clean = if_else(geschlecht == "w", "weiblich", geschlecht))
docs <- load_notes() |>
  mutate(doc_id = row_number()) |>
  left_join(cohort |> select(patient_id, geschlecht_clean, alter), by = "patient_id")

idx_train <- unlist(lapply(split(seq_len(nrow(docs)), docs$verschlechterung),
                           function(rows) sample(rows, floor(0.75 * length(rows)))))
train_docs <- docs[idx_train, ]
test_docs  <- docs[-idx_train, ]

count_terms <- function(df) {
  bind_rows(
    df |> unnest_tokens(term, notiz),
    df |> unnest_tokens(term, notiz, token = "ngrams", n = 2) |> filter(!is.na(term))
  ) |>
    count(doc_id, term, name = "n")
}
n_train <- nrow(train_docs)
vocab <- count_terms(train_docs) |>
  count(term, name = "doc_freq") |>
  filter(doc_freq >= 2, doc_freq <= 0.95 * n_train) |>
  mutate(idf = log((1 + n_train) / (1 + doc_freq)) + 1, col = row_number())

tfidf_matrix <- function(df) {
  weighted <- count_terms(df) |>
    inner_join(vocab, by = "term") |>
    mutate(row = match(doc_id, df$doc_id), value = (1 + log(n)) * idf) |>
    group_by(row) |>
    mutate(value = value / sqrt(sum(value^2))) |>
    ungroup()
  Matrix::sparseMatrix(i = weighted$row, j = weighted$col, x = weighted$value,
                       dims = c(nrow(df), nrow(vocab)), dimnames = list(NULL, vocab$term))
}

y_train <- train_docs$verschlechterung
weights <- ifelse(y_train == 1, n_train / (2 * sum(y_train == 1)),
                  n_train / (2 * sum(y_train == 0)))
model <- glmnet::cv.glmnet(tfidf_matrix(train_docs), y_train, family = "binomial",
                           alpha = 0, weights = weights, type.measure = "auc", nfolds = 5)
predict_notes <- function(fit, df) {
  as.numeric(predict(fit, newx = tfidf_matrix(df), s = "lambda.min", type = "response"))
}
proba <- predict_notes(model, test_docs)
y_test <- test_docs$verschlechterung
cat("=== 1) Train text classifier ===\n")
cat(sprintf("  Baseline test AUC: %.3f\n", auc(y_test, proba)))

# ---- 2) Model persistence ---------------------------------------------------
# Store the model WITH its vocabulary and the package version: a model
# reloaded without the exact preprocessing is a different model.
cat("\n=== 2) Model persistence with saveRDS ===\n")
model_path <- file.path(tempdir(), "klinisches_modell_v1.rds")
saveRDS(list(model = model, vocab = vocab,
             glmnet_version = as.character(packageVersion("glmnet")),
             target = "verschlechterung"), model_path)
loaded <- readRDS(model_path)
cat(sprintf("  Saved: %s  (%.1f KB)\n", basename(model_path), file.size(model_path) / 1024))
cat(sprintf("  Version match: %s  (glmnet %s)\n",
            loaded$glmnet_version == as.character(packageVersion("glmnet")),
            packageVersion("glmnet")))
cat(sprintf("  Reloaded predictions identical: %s\n",
            isTRUE(all.equal(predict_notes(loaded$model, test_docs), proba))))

# ---- 3) Drift monitoring simulation -----------------------------------------
# Concept drift: the text-outcome relationship changes, simulated by flipping
# a growing fraction of test labels.
cat("\n=== 3) Drift monitoring simulation ===\n")
drift_auc <- sapply(DRIFT_FLIP_RATES, function(rate) {
  y_drifted <- y_test
  flip <- sample(length(y_test), floor(rate * length(y_test)))
  y_drifted[flip] <- 1 - y_drifted[flip]
  auc(y_drifted, proba)
})
for (i in seq_along(DRIFT_FLIP_RATES)) {
  cat(sprintf("  Drift %3.0f%%: AUC = %.3f%s\n", 100 * DRIFT_FLIP_RATES[i], drift_auc[i],
              if (drift_auc[i] < 0.70) " <- below threshold!" else ""))
}
cat(sprintf("\n  AUC drop over full drift range: %.3f\n", drift_auc[1] - drift_auc[length(drift_auc)]))
cat("  Without labels in production, only covariate shift is detectable.\n")

# ---- 4) Fairness: subgroup AUC with bootstrap 95%-CI ------------------------
bootstrap_auc_ci <- function(y, p) {
  boot <- replicate(N_BOOTSTRAP, {
    idx <- sample(length(y), replace = TRUE)
    if (length(unique(y[idx])) < 2) NA_real_ else auc(y[idx], p[idx])
  })
  quantile(boot, c(0.025, 0.975), na.rm = TRUE)
}
report_subgroups <- function(group, title) {
  cat(sprintf("\n  -- %s --\n", title))
  groups <- sort(unique(group))
  for (g in groups) {
    mask <- group == g
    ci <- bootstrap_auc_ci(y_test[mask], proba[mask])
    cat(sprintf("  AUC %-12s: %.3f  [95%%-CI %.3f-%.3f]  (n=%d)\n",
                g, auc(y_test[mask], proba[mask]), ci[1], ci[2], sum(mask)))
  }
  # CI of the difference: resample both groups independently.
  m1 <- group == groups[1]; m2 <- group == groups[2]
  diffs <- replicate(N_BOOTSTRAP, {
    i1 <- sample(which(m1), replace = TRUE); i2 <- sample(which(m2), replace = TRUE)
    if (length(unique(y_test[i1])) < 2 || length(unique(y_test[i2])) < 2) NA_real_
    else auc(y_test[i1], proba[i1]) - auc(y_test[i2], proba[i2])
  })
  d_ci <- quantile(diffs, c(0.025, 0.975), na.rm = TRUE)
  cat(sprintf("  AUC difference %s - %s: %+.3f  [95%%-CI %+.3f to %+.3f]\n", groups[1], groups[2],
              auc(y_test[m1], proba[m1]) - auc(y_test[m2], proba[m2]), d_ci[1], d_ci[2]))
}
cat("\n=== 4) Fairness: subgroup AUC with bootstrap 95%-CI ===\n")
cat(sprintf("  Overall AUC: %.3f\n", auc(y_test, proba)))
report_subgroups(test_docs$geschlecht_clean, "Nach Geschlecht")
report_subgroups(if_else(test_docs$alter < 65, "<65 Jahre", ">=65 Jahre"),
                 "Nach Altersgruppe (Schwelle 65 Jahre)")
cat("\n  Overlapping CIs do not prove equality, and non-overlap is a strict criterion.\n")
cat("  The CI of the difference above is the honest test of the gap itself.\n")

cat("\n=== 5) Governance note ===\n")
cat("  Clinical decision support -> MDR 2017/745: typically Class IIa, but under\n")
cat("  Rule 11 software informing decisions that may cause death or irreversible\n")
cat("  deterioration is Class III (serious deterioration -> IIb). EU AI Act: high-risk.\n")
cat("  Not legal advice. This model is for teaching purposes only.\n")
