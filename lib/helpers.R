# Shared R helpers (DRY) used by every module.
#
# Code is English (identifiers, comments); the dataset is a German clinical
# schema, so column names stay German.
#
# Packages: the authoritative, per-module list is in ../DEPENDENCIES-R.md.
# helpers.R itself needs only `readr` (via tidyverse). Individual modules
# declare what they need with require_pkgs() below, which ABORTS (non-zero
# exit) rather than letting a lesson silently produce no output.
#
# Usage:
#   source(file.path("lib", "helpers.R"))
#   require_pkgs("tidyverse", "survival")
#   cohort <- load_cohort()

SEED <- 42

# Determine project root robustly.
#
# `sys.frame(1)$ofile` (the previous approach) silently falls back to
# getwd() whenever the call stack doesn't have a frame 1 (e.g. sourced from
# an interactive console, knitr, or a different working directory), which
# can point .data at the wrong place without any warning. Instead:
#   1. If this file was invoked via `Rscript path/to/helpers.R`, use the
#      command-line argument (--file=) to resolve the script's own path.
#   2. If this file was loaded via `source()`, use sys.frame(1)$ofile /
#      ofile of the sourcing call, but only when it actually resolves to an
#      existing file.
#   3. Otherwise, we do NOT guess via getwd() — .root is left NA and .load()
#      falls through to the URL fallback below.
# Walk upward from a starting directory until we hit the repo root — the only
# directory that contains BOTH `lib/` and `module/`. This is correct at any
# nesting depth (repo root, `lib/`, or a deeply nested `module/<slug>/code/`),
# unlike a fixed number of dirname() calls.
.walk_to_root <- function(start) {
  d <- normalizePath(start, mustWork = FALSE)
  for (i in 1:12) {
    if (dir.exists(file.path(d, "lib")) && dir.exists(file.path(d, "module"))) {
      return(d)
    }
    parent <- dirname(d)
    if (identical(parent, d)) break  # reached filesystem root
    d <- parent
  }
  NA_character_
}

.find_root <- function() {
  # Case 1: Rscript invocation — resolve the script's own directory.
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    if (file.exists(script_path)) {
      root <- .walk_to_root(dirname(normalizePath(script_path)))
      if (!is.na(root)) return(root)
    }
  }

  # Case 2: source()'d — look for an enclosing frame with an $ofile.
  for (i in rev(seq_len(sys.nframe()))) {
    ofile <- tryCatch(sys.frame(i)$ofile, error = function(e) NULL)
    if (!is.null(ofile) && is.character(ofile) && file.exists(ofile)) {
      root <- .walk_to_root(dirname(normalizePath(ofile)))
      if (!is.na(root)) return(root)
    }
  }

  # Case 3: no script path — try walking up from the working directory. This
  # only ever accepts a dir that actually has lib/ + module/, so it cannot
  # silently point at the wrong place; otherwise NA -> URL fallback.
  .walk_to_root(getwd())
}

.root <- .find_root()
.data <- if (!is.na(.root)) file.path(.root, "data") else NA_character_

# --- Published URLs: single source of truth in ../config.json -----------------
# Parsed with a regex rather than jsonlite so that helpers.R has no dependency
# beyond readr. config.json is written by hand and stays flat, so this is safe.
PLACEHOLDER <- "__SET_ME__"

.read_config_value <- function(key) {
  if (is.na(.root)) return(NA_character_)
  cfg <- file.path(.root, "config.json")
  if (!file.exists(cfg)) return(NA_character_)
  txt <- paste(readLines(cfg, warn = FALSE), collapse = "\n")
  m <- regmatches(txt, regexpr(paste0('"', key, '"\\s*:\\s*"[^"]*"'), txt))
  if (length(m) == 0) return(NA_character_)
  sub('^.*:\\s*"([^"]*)"$', "\\1", m)
}

DATA_BASE_URL <- .read_config_value("data_base_url")
DATA_BASE_URL_CONFIGURED <- !is.na(DATA_BASE_URL) && !grepl(PLACEHOLDER, DATA_BASE_URL, fixed = TRUE)

# --- Fail loudly on missing packages -----------------------------------------
# A lesson that cannot run must not exit 0. Before this existed, `Rscript r.R`
# printed "Missing R packages: …" and returned success, so seven modules
# produced zero output while appearing to have run correctly.
require_pkgs <- function(...) {
  pkgs <- c(...)
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    msg <- paste0(
      "\nMissing R packages: ", paste(missing, collapse = ", "), "\n",
      "This lesson cannot run without them. Install with:\n",
      "  install.packages(c(", paste0('"', missing, '"', collapse = ", "), "))\n",
      "See DEPENDENCIES-R.md for the full per-module list.\n"
    )
    # stop() from a script run via Rscript exits with status 1.
    stop(msg, call. = FALSE)
  }
  invisible(TRUE)
}

# Local-first, URL-fallback loader: reads from the local data/ dir when the
# repo root could be resolved and the file exists there (normal course/repo
# usage). Otherwise falls back to DATA_BASE_URL, so this works in
# environments without a local data/ directory (e.g. a bare script sourced
# from an unusual context, or a hosted/Colab-style R session).
.load <- function(name) {
  path <- if (!is.na(.data)) file.path(.data, name) else NA_character_
  if (!is.na(path) && file.exists(path)) {
    return(readr::read_csv(path, show_col_types = FALSE))
  }
  if (!DATA_BASE_URL_CONFIGURED) {
    stop(paste0(
      name, " is missing locally",
      if (!is.na(path)) paste0(" (", path, ")") else "",
      " and no download URL is set.\n",
      "Either run:  python data/generate_data.py\n",
      "or set 'owner'/'repo' and data_base_url in config.json."
    ), call. = FALSE)
  }
  tryCatch(
    readr::read_csv(paste0(DATA_BASE_URL, name), show_col_types = FALSE),
    error = function(e) {
      stop(paste0(
        name, " could not be loaded locally",
        if (!is.na(path)) paste0(" (", path, ")") else "",
        " or from ", DATA_BASE_URL, name,
        ". Run first: python data/generate_data.py, or check config.json / your internet connection."
      ), call. = FALSE)
    }
  )
}

load_cohort <- function() .load("kohorte.csv")
load_vitals <- function() .load("vitalwerte.csv")
load_labs   <- function() .load("labor.csv")
load_notes  <- function() .load("notizen.csv")
