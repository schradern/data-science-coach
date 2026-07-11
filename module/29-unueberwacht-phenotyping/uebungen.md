# Übungen · Unüberwachtes Lernen und Phänotypisierung von Patient:innen

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – k wählen und begründen
Berechne den Silhouettenkoeffizienten für k = 2 bis 8. Bei welchem k ist er am
höchsten? Stimmt das mit dem Elbow-Diagramm überein? Begründe, welches k du in
einem Publikationskontext wählen würdest, und warum die statistische Antwort
allein nicht ausreicht.

## Aufgabe 2 – Auswirkung der Skalierung
Führe k-Means (k = 3) einmal **ohne** Standardisierung auf den Rohdaten durch
und einmal **mit** Standardisierung. Vergleiche die Cluster-Profilmittelwerte
(z. B. SOFA-Score, CRP). Was beobachtest du, und warum verändert Skalierung die
Cluster so stark?

## Aufgabe 3 – Hierarchisches vs. k-Means
Weise jedem/jeder Patient:in sowohl ein k-Means-Label (k = 3) als auch ein
hierarchisches Cluster-Label (Ward, k = 3) zu. Berechne den **Adjusted Rand
Index (ARI)** zwischen beiden Zuordnungen. Berechne zusätzlich, wie viele
Patient:innen bei beiden Verfahren **nicht** in dasselbe Cluster fallen,
Achtung: Cluster-**Nummern** sind zwischen zwei Verfahren arbiträr, ein
naiver Vergleich der rohen Label-Werte (`!=`) ist ohne vorherige Angleichung
(z. B. Ungarischer Algorithmus) nicht aussagekräftig. Was sagt die
Übereinstimmung (oder Nichtübereinstimmung) über die Stabilität der Cluster
aus?

## Aufgabe 4 – Phänotyp-Profil
Erstelle für jeden der drei Cluster eine Profilübersicht: Mittelwerte von
`alter`, `sofa_score`, `crp_mg_l`, `laktat_mmol_l`, `verweildauer_tage`
(Verweildauer) und die 30-Tage-Mortalität (`verstorben_30d`). Vergib
begründete klinische Namen, anhand der Profilwerte, nicht anhand der
rohen Cluster-Nummer. Welche Einschränkungen hat diese Benennung?

## Bonus – UMAP vs. PCA
Projiziere die Daten mit UMAP und mit PCA auf 2 Dimensionen (eingefärbt nach
k-Means-Cluster). Wo unterscheiden sich die Darstellungen? Erkläre, warum UMAP
manchmal klarer trennt, aber weniger interpretierbar ist.
