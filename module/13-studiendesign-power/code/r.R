# Module 13 - Study design, power, and precision.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

suppressPackageStartupMessages(library(tidyverse))

df <- load_cohort()

cat("\n1) Sample size per group for two-sample t-test\n")
for (d in c(0.2, 0.3, 0.5, 0.8)) {
  res <- power.t.test(delta = d, sd = 1, sig.level = 0.05, power = 0.8,
                      type = "two.sample", alternative = "two.sided")
  cat(sprintf("Cohen's d %.1f -> n per group %.1f\n", d, res$n))
}

cat("\n2) Precision of 30-day mortality rate (Wilson score interval)\n")
events <- sum(df$verstorben_30d)
n <- nrow(df)
cat(sprintf("events=%d/%d (%.3f)\n", events, n, events / n))
# correct = FALSE gives the plain Wilson score interval, matching Python's
# statsmodels proportion_confint(method="wilson"). The default correct = TRUE
# would apply a continuity correction (a wider score interval), which is NOT
# what "Wilson-KI" usually means — see the README Stolperstein in section 6.
print(prop.test(events, n, correct = FALSE)$conf.int)

cat("\n3) Events per variable\n")
for (parameters in c(3, 6, 10, 12)) {
  cat(sprintf("%2d parameters -> EPV %.1f\n", parameters, events / parameters))
}
