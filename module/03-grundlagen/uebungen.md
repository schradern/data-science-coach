# Übungen · Programmiergrundlagen in Python und R

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Datentypen identifizieren
Lege in Python und R je eine Variable für folgende klinische Werte an:
CRP-Wert `22.4` (mg/l), SOFA-Score `8`, Aufnahmegrund `"Pneumonie"`,
beatmungspflichtig `True`/`TRUE`.
Gib für jede Variable den Datentyp aus (`type()` in Python, `class()` in R).

## Aufgabe 2 – Vektoren und Grundrechenarten
Gegeben sind die Kreatinin-Werte (mg/dl) von 7 Patient:innen:
`[0.9, 1.4, 2.1, 0.8, 3.7, 1.1, 1.6]`.

a) Berechne Minimum, Maximum und Mittelwert.
b) Welche Patient:innen haben ein Kreatinin > 2,0 mg/dl (mögliche Niereninsuffizienz)?
c) Wie viel Prozent der Patient:innen erfüllen dieses Kriterium?

## Aufgabe 3 – Eigene Funktion schreiben
Schreibe eine Funktion `is_tachycardic(heart_rate)`, die `True`/`TRUE` zurückgibt,
wenn die Herzfrequenz > 100 bpm beträgt.

Wende die Funktion auf folgende Werte an: `[72, 88, 115, 54, 103, 98, 122]`.
Wie viele Patient:innen sind tachykard?

## Aufgabe 4 – Kontrollfluss
Schreibe eine Funktion `assess_map(map_mmhg)`, die den mittleren arteriellen
Druck (MAP) bewertet:

- MAP < 65 → `"Schock-Grenzwert unterschritten"`
- MAP 65–90 → `"Normbereich"`
- MAP > 90 → `"Erhöht"`

Teste mit den Werten `[58, 72, 95, 63, 88]`.

## Aufgabe 5 – DataFrame-Basics mit dem Kohortendatensatz
Lade die Kohorte, z. B. mit `pd.read_csv("{{DATEN}}kohorte.csv")`
(Python) bzw. `read_csv("{{DATEN}}kohorte.csv")` (R, Paket `readr`).

a) Wie viele Patient:innen sind älter als 70 Jahre?
b) Was ist die durchschnittliche Verweildauer bei Patient:innen mit Hypertonie?
c) Erstelle eine neue Spalte `high_crp`, die `True` ist, wenn `crp_mg_l > 50`.
   Wie viele Patient:innen haben ein hohes CRP?

## Bonus
Berechne die **Standardisierung** des Alters (z-Score): `(alter - Mittelwert) / Standardabweichung`.
Füge die z-Werte als neue Spalte `age_z` zum DataFrame hinzu.
Welche Wertebereiche erwartest du theoretisch?
