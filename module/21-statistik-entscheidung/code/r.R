# Module 21 - Statistical test choice in practice.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

suppressPackageStartupMessages(library(tidyverse))

cohort <- load_cohort()
labs <- load_labs()
vitals <- load_vitals()
df <- left_join(cohort, labs, by = "patient_id")

cat("\n1) Independent continuous outcome: lactate by sepsis\n")
sepsis <- df |> filter(aufnahmegrund == "Sepsis") |> pull(laktat_mmol_l) |> na.omit()
other <- df |> filter(aufnahmegrund != "Sepsis") |> pull(laktat_mmol_l) |> na.omit()
print(t.test(sepsis, other, var.equal = FALSE))
print(wilcox.test(sepsis, other, exact = FALSE))

cat("\n2) Paired continuous outcome: MAP day 0 vs day 3\n")
wide <- vitals |>
  select(patient_id, tag, map_mmhg) |>
  pivot_wider(names_from = tag, values_from = map_mmhg) |>
  drop_na(`0`, `3`)
print(summary(wide$`3` - wide$`0`))
print(t.test(wide$`3`, wide$`0`, paired = TRUE))
print(wilcox.test(wide$`3`, wide$`0`, paired = TRUE, exact = FALSE))

cat("\n3) More than two groups: lactate by top four admission reasons\n")
top <- names(sort(table(df$aufnahmegrund), decreasing = TRUE))[1:4]
groups <- lapply(top, function(g) df |> filter(aufnahmegrund == g) |> pull(laktat_mmol_l) |> na.omit())
print(kruskal.test(groups))

cat("\n4) Categorical outcome: active smoking by mortality\n")
tab <- table(aktiv = df$raucherstatus == "aktiv", tod = df$verstorben_30d)
print(tab)
print(chisq.test(tab, correct = FALSE))
print(fisher.test(tab))
