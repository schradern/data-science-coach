# Installs every R package the course uses. See DEPENDENCIES-R.md.
#   Rscript tools/install_r_packages.R
#
# Core = modules 01–10, 12–25, 34. Extras = modules 26–33.
# Extras are optional: the core path runs without them, and each module that
# needs one aborts with an explicit message (never a silent success).

core <- c(
  "tidyverse", "gtsummary", "broom", "survival",
  "DBI", "duckdb", "readxl", "writexl", "jsonlite", "httr2",
  "tidymodels", "pROC", "lme4", "sandwich", "lmtest", "mice", "cmprsk"
)

extras <- c(
  "xgboost", "vip", "pdp", "DALEX", "DALEXtra",
  "randomForestSRC", "timeROC", "umap", "factoextra", "cluster",
  "tidytext", "nnet", "recipes", "ranger", "glmnet"
)

# Rscript runs non-interactively and cannot ask for a mirror; without one,
# install.packages() stops with "trying to use CRAN without setting a mirror".
if (is.null(getOption("repos")) || getOption("repos")[["CRAN"]] %in% c("@CRAN@", "")) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
}

install_missing <- function(pkgs, label) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0) {
    cat(sprintf("%s: all %d packages already installed.\n", label, length(pkgs)))
    return(invisible(character(0)))
  }
  cat(sprintf("%s: installing %d of %d -> %s\n",
              label, length(missing), length(pkgs), paste(missing, collapse = ", ")))
  install.packages(missing)
  still <- missing[!vapply(missing, requireNamespace, logical(1), quietly = TRUE)]
  invisible(still)
}

failed <- c(
  install_missing(core, "Core"),
  install_missing(extras, "Extras")
)

if (length(failed) > 0) {
  cat("\nFailed to install:", paste(failed, collapse = ", "), "\n")
  cat("On Linux these usually need system libraries (libcurl, libxml2, libssl).\n")
  quit(save = "no", status = 1)
}

cat("\nAll R packages available.\n")
