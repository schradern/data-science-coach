# Übungen · Retrieval-Augmented Generation und Leitlinien-Q&A

Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Übung 1: Fehleranalyse in einer RAG-Pipeline
Ein klinisches Assistenzsystem soll Fragen zu Leitlinien beantworten. Ein Arzt stellt die Frage: *„Wie hoch ist die Initialdosis von Amoxicillin bei ambulant erworbener Pneumonie laut Leitlinie?“*
Der Chatbot antwortet: *„Diese Information ist in den bereitgestellten Dokumenten nicht enthalten.“*
Bei der Überprüfung der Vektordatenbank stellst du fest, dass das Leitlinien-PDF sehr wohl einen Absatz zur Dosierung von Amoxicillin enthält.

1. Welche zwei Hauptfehlerquellen (Retrieval-Fehler vs. Generations-Fehler) könnten diese Falschantwort erklären?
2. Wie würdest du systematisch untersuchen, wo der Fehler liegt?

## Übung 2: Prompt-Design zur Halluzinationsvermeidung
Ein LLM soll Leitlinienfragen beantworten. Entwirf einen strukturierten Prompt (System-Prompt + User-Prompt-Template), der folgende Kriterien erfüllt:

1. Das LLM darf nur Fakten aus dem mitgelieferten Kontext nutzen.
2. Wenn die Antwort im Kontext steht, muss der genaue Satz als Zitat angegeben werden.
3. Wenn die Antwort nicht im Kontext steht, muss das LLM mit einem standardisierten Satz antworten und darf keine externen medizinischen Kenntnisse hinzuziehen.

## Übung 3: recall@k und precision@k von Hand berechnen
Ein RAG-System für Leitlinienfragen wird gegen ein kleines Gold-Set geprüft. Für jede Frage ist genau **ein** belegender Gold-Chunk bekannt. Der Retriever gibt pro Frage die **Top-3**-Chunks zurück:

| Frage | Gold-Chunk | Top-3 Treffer (Rang 1–3) |
|---|---|---|
| F1 | C7 | C7, C2, C9 |
| F2 | C4 | C1, C4, C8 |
| F3 | C11 | C3, C6, C2 |
| F4 | C5 | C5, C12, C4 |

1. Bestimme für jede Frage recall@3 und precision@3.
2. Berechne den Makro-Durchschnitt von recall@3 und precision@3 über alle vier Fragen. Berechne zusätzlich recall@1.
3. Für welche Frage kann die Antwort des LLM unmöglich belegt (grounded) sein, und warum?

## Übung 4: Konfidenz-Schwelle als Sicherheitsnetz – Konfusionsmatrix von Hand
Der Grundsatz aus dem Modul: Liegt der Top-1-Ähnlichkeitsscore unter der Schwelle, verweigert das System die Antwort („nicht im Kontext“). Die Schwelle sei `τ = 0,20`. „Positiv“ bedeutet: die Frage ist im Korpus beantwortbar (In-Kontext). Für sechs Testfragen liegen Top-1-Score und wahres Label vor:

| Frage | Top-1-Score | Wahres Label |
|---|---|---|
| Q1 | 0,35 | In-Kontext |
| Q2 | 0,22 | In-Kontext |
| Q3 | 0,14 | In-Kontext |
| Q4 | 0,28 | Außer-Kontext |
| Q5 | 0,10 | Außer-Kontext |
| Q6 | 0,05 | Außer-Kontext |

1. Klassifiziere jede Frage als TP, FP, FN oder TN (das System *antwortet*, wenn Score ≥ τ).
2. Berechne Precision, Recall und F1 der Entscheidung „antworten“.
3. Welche Fehlklassifikation ist klinisch gefährlicher – und in welche Richtung müsste man τ verschieben, um sie zu senken? Welchen Preis zahlt man dafür?
