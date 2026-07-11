# Module 17 - Classic survival analysis.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

suppressPackageStartupMessages(library(survival))

# Real time-to-event columns: fu_zeit_tage (follow-up time) + status
# (1 = death observed, 0 = censored). verweildauer_tage (length of stay) is
# a separate, purely descriptive column and must NOT be used as survival time.

df <- load_cohort()
df$sepsis <- as.integer(df$aufnahmegrund == "Sepsis")

cat("\n1) Kaplan-Meier\n")
km <- survfit(Surv(fu_zeit_tage, status) ~ sepsis, data = df)
print(summary(km, times = c(10, 20, 30)))

cat("\n2) Log-rank\n")
print(survdiff(Surv(fu_zeit_tage, status) ~ sepsis, data = df))

cat("\n3) Cox model\n")
cox <- coxph(Surv(fu_zeit_tage, status) ~ sepsis + alter + sofa_score, data = df)
print(summary(cox)$conf.int)
cat("\nPH assumption check\n")
print(cox.zph(cox))
