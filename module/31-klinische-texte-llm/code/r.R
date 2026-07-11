# Module 31 — Clinical text classification in R (parallel to Python).
#   Rscript module/31-klinische-texte-llm/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; dataset column names stay German.
# R NLP for German text is limited without internet access; this script
# demonstrates the tidytext workflow and logistic regression.

suppressPackageStartupMessages({
  missing_pkgs <- setdiff(c("tidyverse", "tidytext", "tidymodels"), rownames(installed.packages()))
  if (length(missing_pkgs) > 0) {
    message("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
    message("This lesson cannot run without them; nothing was computed.")
    message("Full per-module package list: DEPENDENCIES-R.md")
    message("Install with: install.packages(c('tidyverse', 'tidytext', 'tidymodels'))")
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

# Tokenise and compute tf-idf per document
tidy_tokens <- notes |>
  mutate(doc_id = row_number()) |>
  unnest_tokens(word, notiz) |>
  count(doc_id, word, name = "n") |>
  bind_tf_idf(word, doc_id, n)

# Top tokens by mean tf-idf (rough importance proxy without a model)
top_tokens <- tidy_tokens |>
  group_by(word) |>
  summarise(mean_tfidf = mean(tf_idf), .groups = "drop") |>
  slice_max(mean_tfidf, n = 10)

cat("\n=== 2) Top 10 tokens by mean TF-IDF ===\n")
print(top_tokens)

# Train/test split + logistic regression via tidymodels (document-level)
# We join aggregated tf-idf features back onto the document level.
doc_features <- tidy_tokens |>
  group_by(doc_id) |>
  summarise(
    mean_tfidf  = mean(tf_idf),
    n_tokens    = sum(n),
    .groups = "drop"
  ) |>
  left_join(notes |> mutate(doc_id = row_number()) |>
              select(doc_id, verschlechterung), by = "doc_id") |>
  mutate(verschlechterung = factor(verschlechterung, levels = c(1, 0)))

split  <- initial_split(doc_features, prop = 0.75, strata = verschlechterung)
train  <- training(split)

rec <- recipe(verschlechterung ~ mean_tfidf + n_tokens, data = train) |>
  step_normalize(all_numeric_predictors())

spec   <- logistic_reg() |> set_engine("glm")
wf     <- workflow() |> add_recipe(rec) |> add_model(spec)
fitted <- fit(wf, data = train)

test_pred <- augment(fitted, new_data = testing(split))
auc_val   <- roc_auc(test_pred, truth = verschlechterung,
                     .pred_1, event_level = "first")$.estimate

cat(sprintf("\n=== 3) Held-out test AUC (aggregate features only): %.3f ===\n", auc_val))
cat("Note: Full TF-IDF feature matrices require specialised R packages\n")
cat("(e.g., textrecipes). The Python pipeline in code/python.py is the\n")
cat("primary implementation for this module.\n")
