# Übungen · Verarbeitung klinischer Freitexte mit LLMs

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Pipeline-Leakage demonstrieren
Passe den `TfidfVectorizer` einmal **außerhalb** der Pipeline auf dem gesamten
Datensatz an (`.fit_transform(texts, y)`), dann innerhalb. Vergleiche die
Kreuzvalidierungs-AUC beider Varianten und erkläre den Unterschied.

## Aufgabe 2 – N-Gramme und Stoppwörter
Ändere `ngram_range` auf `(1, 1)` (nur Unigramme). Wie verändert sich die
Test-AUC? Welche klinischen Wortmuster gehen dabei verloren?

## Aufgabe 3 – Tokens interpretieren
Gib die 10 Tokens mit dem positivsten und die 10 mit dem negativsten
Koeffizienten aus. Welche erscheinen klinisch plausibel? Welche überraschen?

## Aufgabe 4 – Schwelle klinisch wählen
Wähle eine Schwelle, bei der die Sensitivität für Verschlechterung mindestens
0,80 beträgt. Wie verändert sich die Spezifität? Wann wäre dieser Kompromiss
klinisch gerechtfertigt?

## Bonus – Regularisierung
Teste `C in [0.01, 0.1, 1, 10]` mit `GridSearchCV` (cv=5, scoring="roc_auc").
Welches C liefert die beste CV-AUC? Erkläre, was starke Regularisierung
(kleines C) mit seltenen, stark assoziierten Tokens macht.
