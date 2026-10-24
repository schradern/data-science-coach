# Musterlösungen · Auswahl der passenden statistischen Methode

## Aufgabe 1 – Testwahl begründen

| Frage | Methode | Warum |
|---|---|---|
| SOFA bei Diabetes ja/nein | Mann-Whitney oder Welch-t | ordinaler, ganzzahliger Punktwert, zwei unabhängige Gruppen; Welch-t vertretbar, wenn der Mittelwert klinisch interessiert |
| MAP Tag 0 vs. Tag 3 | gepaarter t-Test oder Wilcoxon | dieselben Patient:innen werden zweimal gemessen |
| Aufnahmegrund und Mortalität | Chi-Quadrat, ggf. Fisher | kategoriale Variable gegen binären Outcome |
| Laktat in vier Aufnahmegruppen | Kruskal-Wallis | stetig, mehrere Gruppen, Laktat meist rechtsschief |
| Mortalität adjustiert für Alter und SOFA | logistische Regression | binärer Outcome mit Adjustierungsbedarf; Vorsicht: SOFA ist laut DAG (Modul 15) für Diabetes/Sepsis ein Mediator, nicht immer ein Confounder; für den totalen Effekt dieser Expositionen sollte nur Alter adjustiert werden |

## Aufgabe 2 – Gepaarte Daten

```python
import pandas as pd
from scipy import stats

v = pd.read_csv("{{DATEN}}vitalwerte.csv")
wide = v.pivot(index="patient_id", columns="tag", values="map_mmhg").dropna(subset=[0, 3])
diff = wide[3] - wide[0]

print(diff.describe())
print("Schiefe:", round(stats.skew(diff), 2))
print(stats.ttest_rel(wide[3], wide[0]))
print(stats.wilcoxon(wide[3], wide[0]))
```

Wenn die Differenzen ungefähr symmetrisch sind, ist der gepaarte t-Test gut
vertretbar. Bei klarer Schiefe oder Ausreißern ist Wilcoxon robuster. Wichtig:
kein unabhängiger t-Test, weil Tag 0 und Tag 3 pro Patient:in zusammengehören.
Einen Shapiro-Test brauchst du dafür nicht: Bei 408 Paaren wird er schon bei
harmlosen Abweichungen signifikant (siehe README, Abschnitt 4).

In diesen Daten sind die Differenzen symmetrisch (Median 0, IQR −7 bis 7 mmHg,
Schiefe 0,07). Ergebnis: mittlere Differenz −0,01 mmHg (95-%-KI −1,06 bis 1,05),
gepaarter t-Test p = 0,99; Wilcoxon kommt zum selben Schluss (p = 0,90).

## Aufgabe 3 – Mehr als zwei Gruppen

```python
import pandas as pd
from scipy import stats

labs   = pd.read_csv("{{DATEN}}labor.csv")

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(labs, on="patient_id")
top = df["aufnahmegrund"].value_counts().head(4).index
groups = [df.loc[df["aufnahmegrund"].eq(g), "laktat_mmol_l"].dropna() for g in top]

print(stats.kruskal(*groups))
```

Laktat ist rechtsschief und enthält Ausreißer. Kruskal-Wallis ist deshalb die
saubere erste Wahl. Ergebnis: H = 8,66, p = 0,034 (Mediane 1,5 bis 2,1 mmol/l). Post-hoc-Vergleiche sollten vorher geplant und für multiples
Testen korrigiert werden.

## Aufgabe 4 – Kategoriale Daten

```python
import pandas as pd
from scipy import stats

df = pd.read_csv("{{DATEN}}kohorte.csv")
tab = pd.crosstab(df["raucherstatus"].eq("aktiv"), df["verstorben_30d"])
chi2, p, dof, expected = stats.chi2_contingency(tab, correction=False)
print(tab)
print(expected)
print("Chi-square:", chi2, p)
print("Fisher:", stats.fisher_exact(tab))
```

Wenn erwartete Zellhäufigkeiten unter 5 liegen, ist Fisher exakt die konservative
Wahl. Bei ausreichend großen erwarteten Häufigkeiten ist Chi-Quadrat gut lesbar.
Hier ist die kleinste erwartete Häufigkeit 17, also Chi-Quadrat: χ² = 0,36,
p = 0,55; Fisher liefert dasselbe p (0,55, OR 1,19).

## Aufgabe 5 – Reportingsatz

Beispiel: „Da Laktat rechtsschief verteilt war, verglichen wir Sepsis vs.
Nicht-Sepsis mit dem Mann-Whitney-U-Test und berichteten zusätzlich das
rang-biseriale r. Das Laktat war bei Sepsis leicht höher (Median 2,0 [IQR 1,35–3,1]
vs. 1,8 [1,0–2,8] mmol/l; r = 0,15; p = 0,035); ein Unterschied von 0,2 mmol/l ist
klinisch eher klein.“ Für die Unsicherheit ergänzt du ein 95-%-KI, z. B. per
Bootstrap für die Mediandifferenz.
