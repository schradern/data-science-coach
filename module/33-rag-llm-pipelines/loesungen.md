# Musterlösungen · Retrieval-Augmented Generation und Leitlinien-Q&A

## Übung 1: Fehleranalyse in einer RAG-Pipeline

1. **Hauptfehlerquellen:**
    * **Retrieval-Fehler:** Die Vektorsuche hat versagt. Trotz Vorhandenseins der Information im Dokument war die semantische Ähnlichkeit zwischen Frage und Textchunk zu gering, oder die Chunks waren so ungünstig geschnitten, dass der relevante Absatz nicht unter den Top-K-Ergebnissen landete. Der Kontext blieb leer.
    * **Generations-Fehler:** Das Retrieval war erfolgreich und der Absatz wurde an den Prompt übergeben. Das LLM hat den Text jedoch nicht korrekt interpretiert oder aufgrund eines zu restriktiven System-Prompts („Antworte nur bei 100%iger Übereinstimmung“) die Antwort verweigert.

2. **Systematische Untersuchung:**
    * **Schritt 1:** Gib den Roh-Output des Retrievers aus (z. B. die Top 3 ähnlichsten Text-Chunks zur Frage). 
     * *Ergebnis A:* Der korrekte Absatz ist nicht dabei. -> Fehler liegt beim Retrieval (Chunk-Größe anpassen, anderes Embedding-Modell nutzen, Synonyme in der Frage ergänzen).
     * *Ergebnis B:* Der Absatz ist dabei. -> Fehler liegt bei der Generation.
    * **Schritt 2 (falls Ergebnis B):** Überprüfe den Prompt und die Modellparameter (z. B. Temperatur senken, System-Prompt lockern).

---

## Übung 2: Prompt-Design zur Halluzinationsvermeidung

### Lösungsvorschlag
```
[SYSTEM PROMPT]
Du bist ein streng regelbasierter medizinischer Assistent. Deine Aufgabe ist es, Fragen ausschließlich unter Verwendung des bereitgestellten "KONTEXT" zu beantworten. 

Folge diesen Regeln penibel:

1. Verwende KEIN medizinisches Außenwissen. Wenn die Antwort nicht direkt im KONTEXT steht, antworte exakt mit: "Information in den bereitgestellten Dokumenten nicht verfügbar."
2. Wenn du eine Antwort formulierst, füge am Ende der Antwort den exakten Satz aus dem KONTEXT als Zitat in Anführungszeichen an (z. B. Zitat: "Noradrenalin ist der Vasopressor der ersten Wahl").
3. Spekuliere nicht und interpretiere keine unklaren Sätze.

[KONTEXT]
{retrieved_text_chunks}

---

[USER FRAGE]
{user_query}
```

---

## Übung 3: recall@k und precision@k von Hand berechnen

1. **Pro Frage** (es existiert genau 1 relevanter Chunk; `precision@3 = Treffer im Top-3 / 3`):

    - F1: Gold C7 auf Rang 1 → gefunden. `recall@3 = 1`; `precision@3 = 1/3`.
    - F2: Gold C4 auf Rang 2 → gefunden. `recall@3 = 1`; `precision@3 = 1/3`.
    - F3: Gold C11 nicht unter den Top-3 → verfehlt. `recall@3 = 0`; `precision@3 = 0`.
    - F4: Gold C5 auf Rang 1 → gefunden. `recall@3 = 1`; `precision@3 = 1/3`.

2. **Makro-Durchschnitte:**

    - `Makro-recall@3 = (1 + 1 + 0 + 1) / 4 = 0,75`
    - `Makro-precision@3 = (1/3 + 1/3 + 0 + 1/3) / 4 = 1 / 4 = 0,25`
    - `recall@1` (Gold auf Rang 1): F1 ✓, F2 ✗ (Rang 2), F3 ✗, F4 ✓ → `2 / 4 = 0,50`

    (Die maximale `precision@3` ist hier `1/3 ≈ 0,33`, weil je Frage nur ein Chunk relevant ist – genau der Wert aus dem README.)

3. **Grounding:** Bei **F3** liegt der Gold-Chunk (C11) nicht im übergebenen Kontext. Das LLM bekommt die belegende Textstelle nie zu sehen; jede inhaltliche Antwort wäre entweder eine Halluzination oder aus Vorwissen geraten – **nicht grounded**. Korrekt wäre hier die Ausgabe „Information nicht im Kontext enthalten“. Das ist ein reiner **Retrieval-Fehler** (vgl. Übung 1): Der beste Prompt kann eine fehlende Quelle nicht retten.

---

## Übung 4: Konfidenz-Schwelle als Sicherheitsnetz – Konfusionsmatrix von Hand

1. **Klassifikation** (System *antwortet*, wenn Score ≥ 0,20):

    - Q1 (0,35, In-Kontext): antwortet, korrekt → **TP**
    - Q2 (0,22, In-Kontext): antwortet, korrekt → **TP**
    - Q3 (0,14, In-Kontext): verweigert, obwohl beantwortbar → **FN**
    - Q4 (0,28, Außer-Kontext): antwortet, obwohl keine Quelle → **FP**
    - Q5 (0,10, Außer-Kontext): verweigert, korrekt → **TN**
    - Q6 (0,05, Außer-Kontext): verweigert, korrekt → **TN**

    Also `TP = 2`, `FP = 1`, `FN = 1`, `TN = 2`.

2. **Kennzahlen:**

    - `Precision = TP / (TP + FP) = 2 / 3 ≈ 0,67`
    - `Recall = TP / (TP + FN) = 2 / 3 ≈ 0,67`
    - `F1 = 2 · P · R / (P + R) = 2 · (2/3) · (2/3) / (4/3) = 2/3 ≈ 0,67`

3. **Gefährlichere Fehlklassifikation:** Der **False Positive Q4**. Das System übergibt einen Außer-Kontext-Chunk als vermeintliche Quelle an das LLM, das darauf eine scheinbar belegte, in Wahrheit ungegroundete Antwort formulieren kann. Um FP zu senken, müsste man τ **anheben** (z. B. auf 0,30) – dann fällt Q4 unter die Schwelle. Der Preis: Q2 (Score 0,22, echt beantwortbar) würde ebenfalls verweigert – ein zusätzlicher FN. Precision (Sicherheit) und Recall (Abdeckung) stehen im Zielkonflikt; die Schwelle muss **out-of-sample** kalibriert werden, nicht auf denselben Fragen, an denen man sie bewertet (Stolperstein „Zirkuläre Kalibrierung“, README Abschnitt 4).
