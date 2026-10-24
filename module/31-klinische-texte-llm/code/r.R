# Module 31 – Clinical text classification in R (parallel to Python).
#   Rscript module/31-klinische-texte-llm/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; dataset column names stay German.
# Same model as python.py: TF-IDF over unigrams and bigrams, fitted on the
# training notes only, then a penalised logistic regression (glmnet).

suppressPackageStartupMessages({
  missing_pkgs <- setdiff(c("tidyverse", "tidytext", "tidymodels", "glmnet"), rownames(installed.packages()))
  if (length(missing_pkgs) > 0) {
    message("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
    message("This lesson cannot run without them; nothing was computed.")
    message("Full per-module package list: DEPENDENCIES-R.md")
    message("Install with: install.packages(c('tidyverse', 'tidytext', 'tidymodels', 'glmnet'))")
    quit(save = "no", status = 1)
  }
  library(tidyverse)
  library(tidytext)
  library(tidymodels)
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

notes <- load_notes()

cat("=== 1) Dataset overview ===\n")
cat(sprintf("  %d notes | %.1f%% positive\n",
            nrow(notes), 100 * mean(notes$verschlechterung)))

docs <- notes |> mutate(doc_id = row_number())

# ---- 2) Split FIRST, then learn vocabulary and IDF on the training notes ----
# Like python.py's Pipeline: a vocabulary or IDF computed on all notes would
# let the test notes leak into the features.
split <- initial_split(docs, prop = 0.75, strata = verschlechterung)
train_docs <- training(split)
test_docs  <- testing(split)

# Unigrams + bigrams, as in python.py (ngram_range = (1, 2)).
count_terms <- function(df) {
  bind_rows(
    df |> unnest_tokens(term, notiz),
    df |> unnest_tokens(term, notiz, token = "ngrams", n = 2) |> filter(!is.na(term))
  ) |>
    count(doc_id, term, name = "n")
}
train_counts <- count_terms(train_docs)
test_counts  <- count_terms(test_docs)

# Vocabulary and IDF from the training notes only: min_df = 2, max_df = 0.95,
# smoothed IDF and sublinear tf, the same settings as sklearn's TfidfVectorizer.
n_train <- nrow(train_docs)
vocab <- train_counts |>
  count(term, name = "doc_freq") |>
  filter(doc_freq >= 2, doc_freq <= 0.95 * n_train) |>
  mutate(idf = log((1 + n_train) / (1 + doc_freq)) + 1, col = row_number())

tfidf_matrix <- function(counts, doc_ids) {
  weighted <- counts |>
    inner_join(vocab, by = "term") |>
    mutate(row = match(doc_id, doc_ids), value = (1 + log(n)) * idf) |>
    group_by(row) |>
    mutate(value = value / sqrt(sum(value^2))) |>   # L2 row normalisation
    ungroup()
  Matrix::sparseMatrix(i = weighted$row, j = weighted$col, x = weighted$value,
                       dims = c(length(doc_ids), nrow(vocab)),
                       dimnames = list(NULL, vocab$term))
}
X_train <- tfidf_matrix(train_counts, train_docs$doc_id)
X_test  <- tfidf_matrix(test_counts, test_docs$doc_id)
y_train <- train_docs$verschlechterung
y_test  <- test_docs$verschlechterung
cat(sprintf("\n=== 2) Train/test split: %d / %d notes | vocabulary (train only): %d unigrams/bigrams ===\n",
            nrow(train_docs), nrow(test_docs), nrow(vocab)))

# ---- 3) Ridge-penalised logistic regression on the full TF-IDF matrix -------
# class weights 'balanced' as in python.py; lambda chosen by 5-fold CV (AUC)
# inside the training set.
weights <- ifelse(y_train == 1, n_train / (2 * sum(y_train == 1)),
                  n_train / (2 * sum(y_train == 0)))
cv_fit <- glmnet::cv.glmnet(X_train, y_train, family = "binomial", alpha = 0,
                            weights = weights, type.measure = "auc", nfolds = 5)
cat(sprintf("\n=== 3) 5-fold CV-AUC (training set only): %.3f ===\n", max(cv_fit$cvm)))

proba <- as.numeric(predict(cv_fit, newx = X_test, s = "lambda.min", type = "response"))
test_auc <- roc_auc_vec(factor(y_test, levels = c(1, 0)), proba, event_level = "first")
cat(sprintf("\n=== 4) Held-out test AUC: %.3f ===\n", test_auc))

# ---- 5) Most predictive tokens ----------------------------------------------
coefs <- as.matrix(coef(cv_fit, s = "lambda.min"))[-1, 1]
top <- head(coefs[order(abs(coefs), decreasing = TRUE)], 15)
cat("\n=== 5) Most predictive tokens (largest |log-odds coefficient|) ===\n")
print(tibble(token = names(top), koeffizient = round(unname(top), 3)) |>
        arrange(desc(koeffizient)), n = 15)
cat("\nA bag-of-words model ignores word order and negation context.\n")
cat("R and python.py draw different train/test splits and tune the penalty differently\n")
cat("(cv.glmnet here, C = 1 there): with 126 test notes, AUC differences of several\n")
cat("points between the two tracks are sampling noise, not a better model.\n")
