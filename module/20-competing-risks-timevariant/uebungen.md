# Übungen · Konkurrierende Risiken und zeitabhängige Cox-Modelle

## Übung 1: Immortal Time Bias erkennen (Methoden-Kritik)
Lies den folgenden Auszug aus einer fiktiven klinischen Publikation:
> *„Wir untersuchten retrospektiv, ob der Einsatz von extrakorporaler Membranoxygenierung (ECMO) bei Patient:innen mit schwerem ARDS das 30-Tage-Überleben verbessert. Alle Patient:innen, die im Verlauf ihres Intensivaufenthalts eine ECMO erhielten, wurden der ECMO-Gruppe zugeordnet (N=45). Alle anderen bildeten die Kontrollgruppe (N=150). Die Zeitmessung startete am Tag der ICU-Aufnahme.“*

1. Welcher methodische Fehler liegt hier vor?
2. Warum führt dieser Fehler wahrscheinlich zu einer massiven Überschätzung der ECMO-Wirksamkeit?
3. Wie könnte man diesen Fehler mathematisch korrigieren?

## Übung 2: Daten ins Start-Stop-Format überführen (Konzept)
Ein:e Patient:in (ID 42) wird an Tag 0 auf die Intensivstation aufgenommen.

- An Tag 4 wird eine Lungenentzündung (Pneumonie) diagnostiziert.
- An Tag 10 verstirbt der/die Patient:in.
Du willst den Einfluss der Pneumonie (als zeitabhängige Kovariate) auf die Mortalität analysieren.

1. Erstelle die Tabellenzeilen für Patient:in 42 im Start-Stop-Format.
2. Welche Spalten sind zwingend erforderlich?

## Übung 3: 1−KM vs. CIF von Hand berechnen
In einer kleinen Intensivkohorte von 20 Patient:innen konkurrieren zwei Ereignisse: **Tod im Krankenhaus** (Zielereignis) und **lebende Entlassung** (Konkurrenzereignis). Am Tag 5 endet die Beobachtung (administrative Zensierung). Die Ereignisse verteilen sich wie folgt:

| Tag | Risikoset nⱼ | Todesfälle | Entlassungen |
|---|---|---|---|
| 1 | 20 | 2 | 0 |
| 2 | 18 | 0 | 4 |
| 3 | 14 | 2 | 0 |
| 4 | 12 | 0 | 6 |
| 5 | 6 | 1 | 0 |

(Nach Tag 5 sind die verbleibenden 5 Patient:innen zensiert.)

1. Berechne das **naive 1−Kaplan-Meier**-Sterberisiko an Tag 5, indem du Entlassungen als Zensierungen behandelst (KM nur über die Todes-Zeitpunkte).
2. Berechne die **korrekte Cumulative Incidence Function (CIF)** für den Tod an Tag 5 nach Aalen-Johansen: `CIF(t) = Σ S(tⱼ₋₁) · (d_Tod,ⱼ / nⱼ)`, wobei `S` die Gesamtüberlebensfunktion ist (jedes Ereignis – Tod *oder* Entlassung – zählt).
3. Vergleiche beide Werte. Warum überschätzt das naive 1−KM?

## Übung 4: Warum sich konkurrierende CIFs korrekt zu 1 addieren
Am Ende der 30-tägigen Beobachtung teilt sich die Kurskohorte (500 Patient:innen) in drei sich ausschließende Endzustände: **78 verstorben**, **315 lebend entlassen**, **107 zensiert**. Nimm vereinfachend an, alle 107 seien an Tag 30 noch im Krankenhaus (in Wahrheit fallen 18 von ihnen schon vorher aus der Beobachtung). Ohne Zensierung vor Tag 30 sind die CIF-Werte an Tag 30 einfach die beobachteten Anteile.

1. Berechne `CIF_Tod(30)`, `CIF_Entlassung(30)` und den Anteil noch im Krankenhaus (`S(30)`). Zeige, dass sie sich zu 1 addieren.
2. Ein:e Kolleg:in berechnet stattdessen 1−KM getrennt für Tod (27,7 %) und für Entlassung (jeweils das andere Ereignis als Zensierung). Warum können diese beiden Zahlen zusammen die Grenze von 100 % sprengen, während die CIFs es nie tun?
3. Welche Zahl gibst du einer/einem Ärzt:in auf die Frage „Wie hoch ist die Wahrscheinlichkeit, dass diese Patientin binnen 30 Tagen im Krankenhaus verstirbt?“
