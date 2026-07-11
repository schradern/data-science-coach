# Übungen · Datenbereinigung und Datentransformation

Nutze den Kohorten-, Labor- und Vitalwerte-Datensatz. Versuche jede Aufgabe
zuerst in **Python**, dann in **R**. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Fehlende Werte
Wie viele Patient:innen haben einen fehlenden `bmi`? Wie viel Prozent der Kohorte
ist das? Entscheide begründet: löschen oder behalten?

## Aufgabe 2 – Filtern und Ableiten
Erzeuge eine Tabelle nur der Patient:innen mit `sofa_score >= 6`. Füge eine Spalte
`risikogruppe` hinzu, die `"hoch"` ist, wenn `alter >= 70`, sonst `"mittel"`.

## Aufgabe 3 – Gruppieren
Berechne pro `raucherstatus` die mittlere Verweildauer und die 30-Tage-Mortalität.
Welche Gruppe sticht heraus?

## Aufgabe 4 – Join
Verbinde `kohorte` mit `labor`. Wie hoch ist das mediane Kreatinin bei
Patient:innen mit vs. ohne Diabetes?

## Aufgabe 5 – Long ↔ Wide
Forme die Vitalwerte so um, dass je Patient:in eine Spalte je Tag für `map_mmhg`
entsteht. Wie viele Patient:innen haben einen Wert an Tag 3?

## Bonus
Berechne pro Patient:in die **mittlere Herzfrequenz** über alle gemessenen Tage
(aus dem long format) und hänge sie an die Kohorte an.
