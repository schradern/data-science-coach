"""Konzeptdiagramme des Kurses – ein Generator, ein Stil.

Erzeugt minimalistische, konsistente SVG-Diagramme in die `assets/`-Ordner der
Module. Eine einzige Quelle der Wahrheit für Farben, Typografie und Primitive:
Stil global ändern = hier `THEME` anpassen und neu erzeugen:

    python lib/diagrams.py

Bewusst zurückhaltend: viel Weißraum, Haarlinien, eine Akzentfarbe, semantische
Farben (Rot/Bernstein/Grün) nur, wo sie Bedeutung tragen. Die SVGs bringen eine
helle Bildfläche mit – wie die Matplotlib-Plots des Kurses – und werden als
`<img>` eingebunden.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

# --- Ein Stil für alles -----------------------------------------------------
THEME = dict(
    ink="#16181C", ink2="#464A50", mut="#6B7178", faint="#9AA0A8",
    rule="#E7E9ED", hair="#F0F1F4", soft="#F7F8FA", card="#FFFFFF", plate="#FFFFFF",
    acc="#2A5C8A", acc2="#1E456B", acct="#EAF1F8",
    death="#B5482E", deatht="#FBEEEA",
    warn="#9A5B12", warnt="#FBF3E6",
    ok="#2C7A54", okt="#E9F4EE",
)
SANS = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace"
T = THEME


# --- Primitive --------------------------------------------------------------
def txt(x, y, s, size=13, fill=T["ink2"], w=None, anchor=None, mono=False,
        deco=None, ls=None, opacity=None):
    a = [f'x="{x}"', f'y="{y}"', f'font-size="{size}"', f'fill="{fill}"',
         f'font-family="{MONO if mono else SANS}"']
    if w: a.append(f'font-weight="{w}"')
    if anchor: a.append(f'text-anchor="{anchor}"')
    if deco: a.append(f'text-decoration="{deco}"')
    if ls: a.append(f'letter-spacing="{ls}"')
    if opacity is not None: a.append(f'opacity="{opacity}"')
    return f'<text {" ".join(a)}>{escape(s)}</text>'


def rect(x, y, w, h, r=12, fill=T["card"], stroke=None, sw=1, dash=None):
    a = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'rx="{r}"', f'fill="{fill}"']
    if stroke: a.append(f'stroke="{stroke}"'); a.append(f'stroke-width="{sw}"')
    if dash: a.append(f'stroke-dasharray="{dash}"')
    return f'<rect {" ".join(a)}/>'


def line(x1, y1, x2, y2, stroke=T["rule"], sw=1, dash=None, cap="round"):
    a = [f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
         f'stroke="{stroke}"', f'stroke-width="{sw}"', f'stroke-linecap="{cap}"']
    if dash: a.append(f'stroke-dasharray="{dash}"')
    return f'<line {" ".join(a)}/>'


def pill(x, y, w, h, label, fill, fg, size=10.5, mono=False):
    return (rect(x, y, w, h, r=h / 2, fill=fill)
            + txt(x + w / 2, y + h / 2 + size * 0.36, label, size=size, fill=fg,
                  w=600, anchor="middle", mono=mono))


def arrow(x1, y, x2, color=T["mut"], sw=1.6):
    """Waagerechter Pfeil mit kleiner Spitze."""
    head = (f'<path d="M{x2},{y} l-7,-4 M{x2},{y} l-7,4" stroke="{color}" '
            f'stroke-width="{sw}" fill="none" stroke-linecap="round"/>')
    return line(x1, y, x2, y, stroke=color, sw=sw) + head


def canvas(w, h, body, aria):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="{SANS}" role="img" aria-label="{escape(aria)}">'
            f'{rect(0, 0, w, h, r=16, fill=T["plate"])}{body}</svg>\n')


def eyebrow(x, y, s):
    return txt(x, y, s.upper(), size=10.5, fill=T["faint"], w=700, ls="0.14em")


# --- 05 · Relationales Datenmodell ------------------------------------------
def datenmodell() -> str:
    b = [eyebrow(40, 40, "Datenmodell")]
    W = 196
    row_h = 26

    def table(x, y, name, hint, cols, keyrow=0):
        parts = [rect(x, y, W, 34 + row_h * len(cols), r=12, fill=T["card"], stroke=T["rule"])]
        parts.append(txt(x + 16, y + 22, name, size=13.5, fill=T["ink"], w=600))
        parts.append(txt(x + W - 14, y + 21, hint, size=10, fill=T["faint"], anchor="end"))
        parts.append(line(x + 14, y + 34, x + W - 14, y + 34, stroke=T["hair"]))
        for i, (col, tag) in enumerate(cols):
            cy = y + 34 + row_h * i
            key = i == keyrow
            if key:
                parts.append(rect(x + 1, cy, W - 2, row_h, r=0, fill=T["acct"]))
                parts.append(rect(x + 1, cy, 3, row_h, r=0, fill=T["acc"]))
            parts.append(txt(x + 16, cy + 17, col, size=12,
                             fill=T["ink"] if key else T["ink2"], w=600 if key else None,
                             mono=True))
            if tag:
                parts.append(txt(x + W - 14, cy + 17, tag, size=9, fill=T["acc"],
                                 w=700, anchor="end"))
            if i < len(cols) - 1:
                parts.append(line(x + 14, cy + row_h, x + W - 14, cy + row_h, stroke=T["hair"]))
        return "".join(parts), 34 + row_h * len(cols)

    kx, ky = 44, 118
    ktbl, kh = table(kx, ky, "kohorte", "1 Zeile / Patient:in",
                     [("patient_id", "PK"), ("alter", ""), ("aufnahmegrund", ""),
                      ("sofa_score", ""), ("verstorben_30d", "")])
    rx = 540
    specs = [(64, "vitalwerte", [("patient_id", "FK"), ("herzfrequenz", ""), ("mad_mmhg", ""), ("spo2", "")]),
             (232, "labor", [("patient_id", "FK"), ("laktat_mmol_l", ""), ("crp_mg_l", ""), ("leukozyten", "")]),
             (400, "notizen", [("patient_id", "FK"), ("notiztext", "")])]
    right = []
    key_y = ky + 34 + row_h / 2      # patient_id-Zeilenmitte kohorte
    for (yy, nm, cols) in specs:
        tbl, _ = table(rx, yy, nm, "viele / Patient:in", cols)
        right.append(tbl)
        ty = yy + 34 + row_h / 2      # patient_id-Zeilenmitte rechts
        b.append(f'<path d="M{kx+W},{key_y} C{kx+W+120},{key_y} {rx-120},{ty} {rx},{ty}" '
                 f'stroke="{T["acc"]}" stroke-width="1.5" fill="none"/>')
        b.append(f'<path d="M{rx},{ty} l-10,-5 M{rx},{ty} l-10,0 M{rx},{ty} l-10,5" '
                 f'stroke="{T["acc"]}" stroke-width="1.5" fill="none" stroke-linecap="round"/>')
        b.append(txt(rx - 18, ty - 8, "∞", size=12, fill=T["acc"], w=700, anchor="end"))
    b.append(f'<circle cx="{kx+W}" cy="{key_y}" r="3.4" fill="{T["acc"]}"/>')
    b.append(txt(kx + W + 12, key_y - 8, "1", size=12, fill=T["acc"], w=700))
    b.append(ktbl)
    b.extend(right)
    # Legende
    ly = ky + kh + 34
    b.append(txt(kx, ly, "PK", size=11, fill=T["acc"], w=700))
    b.append(txt(kx + 26, ly, "Primärschlüssel", size=11, fill=T["mut"]))
    b.append(txt(kx, ly + 20, "FK", size=11, fill=T["acc"], w=700))
    b.append(txt(kx + 26, ly + 20, "Fremdschlüssel · patient_id verbindet alles", size=11, fill=T["mut"]))
    return canvas(760, 500, "".join(b),
                  "Datenmodell: kohorte hat eine Zeile pro Patient:in (Primärschlüssel patient_id); "
                  "vitalwerte, labor und notizen verweisen über den Fremdschlüssel patient_id zurück (1 zu unendlich).")


# --- 05 · JOIN-Typen --------------------------------------------------------
def join_typen() -> str:
    pats = [("Pat. 1", "Sepsis", "2.1"), ("Pat. 2", "Pneumonie", None),
            ("Pat. 3", "Herzinsuff.", "1.8"), ("Pat. 4", "Sepsis", "3.4"),
            ("Pat. 5", "COPD", None)]
    rw, rh, gap = 336, 40, 8
    y0 = 70

    def panel(x, kw, kwcolor, sub, mode):
        p = [txt(x, 40, kw, size=14, fill=kwcolor, w=700, mono=True),
             txt(x + (128 if mode == "inner" else 118), 40, sub, size=11.5, fill=T["mut"])]
        for i, (name, diag, lak) in enumerate(pats):
            y = y0 + i * (rh + gap)
            has = lak is not None
            if mode == "inner" and not has:
                p.append(rect(x, y, rw, rh, r=10, fill=T["deatht"]))
                p.append(txt(x + 16, y + 25, f"{name}", size=12.5, fill=T["death"], w=600, deco="line-through"))
                p.append(txt(x + 74, y + 25, f"· {diag}", size=12.5, fill=T["death"], deco="line-through", opacity=0.8))
                p.append(txt(x + rw - 16, y + 25, "fällt weg", size=12, fill=T["death"], w=600, anchor="end"))
            elif mode == "left" and not has:
                p.append(rect(x, y, rw, rh, r=10, fill=T["warnt"]))
                p.append(txt(x + 16, y + 25, name, size=12.5, fill=T["ink"], w=600))
                p.append(txt(x + 74, y + 25, f"· {diag}", size=12.5, fill=T["mut"]))
                p.append(txt(x + rw - 16, y + 25, "laktat = NULL", size=11.5, fill=T["warn"], w=700, anchor="end", mono=True))
            else:
                p.append(rect(x, y, rw, rh, r=10, fill=T["card"], stroke=T["rule"]))
                p.append(txt(x + 16, y + 25, name, size=12.5, fill=T["ink"], w=600))
                p.append(txt(x + 74, y + 25, f"· {diag}", size=12.5, fill=T["mut"]))
                p.append(txt(x + rw - 16, y + 25, f"laktat {lak}", size=12, fill=T["acc2"], w=600, anchor="end", mono=True))
        fy = y0 + len(pats) * (rh + gap) + 16
        p.append(line(x, fy, x + rw, fy, stroke=T["rule"]))
        return "".join(p), fy

    left, fy = panel(24, "INNER JOIN", T["death"], "nur mit Laborwert", "inner")
    right, _ = panel(400, "LEFT JOIN", T["acc"], "alle bleiben", "left")
    b = [left, right,
         line(380, 24, 380, fy - 6, stroke=T["rule"], dash="2 5"),
         txt(24, fy + 26, "3 von 5 Zeilen", size=13, fill=T["ink"], w=600),
         txt(24, fy + 45, "2 Patient:innen fehlen, unbemerkt.", size=11.5, fill=T["mut"]),
         txt(400, fy + 26, "5 von 5 Zeilen", size=13, fill=T["ink"], w=600),
         txt(400, fy + 45, "Fehlende Werte sichtbar als NULL.", size=11.5, fill=T["mut"])]
    return canvas(760, fy + 60, "".join(b),
                  "INNER JOIN gegen LEFT JOIN: Der INNER JOIN behält nur die drei Patient:innen mit Laborwert "
                  "(zwei fallen unbemerkt weg); der LEFT JOIN behält alle fünf, fehlende Werte erscheinen als NULL.")


# --- 03 · Anatomie eines DataFrame ------------------------------------------
def dataframe_anatomie() -> str:
    cols = [("patient_id", "int", "1042"), ("alter", "int", "67"),
            ("aufnahmegrund", "str", "Sepsis"), ("laktat", "float", "3.4"),
            ("verstorben_30d", "bool", "True")]
    rows = [["1042", "67", "Sepsis", "3.4", "True"],
            ["1043", "54", "Pneumonie", "1.6", "False"],
            ["1044", "72", "Herzinsuff.", "2.0", "False"]]
    x0, y0 = 150, 96
    cw, hh, rh = 112, 52, 38
    tw = cw * len(cols)
    b = [eyebrow(40, 40, "Anatomie eines DataFrame")]
    # Kopf mit dtype-Pillen
    b.append(rect(x0, y0, tw, hh, r=0, fill=T["soft"]))
    for j, (name, dt, _) in enumerate(cols):
        cx = x0 + j * cw
        b.append(txt(cx + 14, y0 + 22, name, size=11.5, fill=T["ink"], w=600, mono=True))
        b.append(pill(cx + 14, y0 + 30, 42, 15, dt, T["acct"], T["acc2"], size=9.5, mono=True))
        if j: b.append(line(cx, y0, cx, y0 + hh + rh * len(rows), stroke=T["hair"]))
    # Datenzeilen
    for i, r in enumerate(rows):
        ry = y0 + hh + i * rh
        if i: b.append(line(x0, ry, x0 + tw, ry, stroke=T["hair"]))
        for j, v in enumerate(r):
            b.append(txt(x0 + j * cw + 14, ry + 24, v, size=11.5, fill=T["ink2"], mono=True))
    # Rahmen
    b.append(rect(x0, y0, tw, hh + rh * len(rows), r=10, fill="none", stroke=T["rule"]))
    # Spalten-Hervorhebung (eine Variable)
    hx = x0 + 3 * cw
    b.append(rect(hx, y0, cw, hh + rh * len(rows), r=8, fill="none", stroke=T["acc"], sw=1.6))
    b.append(txt(hx + cw / 2, y0 - 12, "eine Spalte = eine Variable", size=11, fill=T["acc"], w=600, anchor="middle"))
    # Zeilen-Label (eine Beobachtung)
    ry = y0 + hh + rh + rh / 2
    b.append(arrow(150 - 14, ry, x0 + 6, color=T["ok"]))
    b.append(txt(40, ry - 8, "eine Zeile =", size=11, fill=T["ok"], w=600))
    b.append(txt(40, ry + 8, "eine Beobachtung", size=11, fill=T["ok"], w=600))
    return canvas(760, 320, "".join(b),
                  "Ein DataFrame ist eine Tabelle: jede Spalte ist eine Variable mit einem Datentyp "
                  "(int, str, float, bool), jede Zeile eine Beobachtung (eine Patient:in).")


# --- 04 · Von der Rohdatei zum DataFrame ------------------------------------
def einlesen_pipeline() -> str:
    b = [eyebrow(40, 40, "Von der Rohdatei zum DataFrame")]
    cy = 110
    # Rohdatei
    b.append(rect(40, cy - 40, 190, 118, r=12, fill=T["soft"]))
    b.append(txt(56, cy - 18, "labor.csv", size=12.5, fill=T["ink"], w=600, mono=True))
    for k, ln in enumerate(["patient_id;laktat", "1042;3,4", "1043;1,6"]):
        b.append(txt(56, cy + 6 + k * 20, ln, size=11, fill=T["mut"], mono=True))
    # Parser
    px = 300
    b.append(rect(px, cy - 40, 200, 118, r=12, fill=T["card"], stroke=T["rule"]))
    b.append(txt(px + 100, cy - 18, "einlesen", size=12.5, fill=T["acc2"], w=600, anchor="middle"))
    for k, (lab) in enumerate(["Trennzeichen  ;", "Encoding  UTF-8", "Dezimal  ,", "Header  Zeile 1"]):
        yy = cy + 2 + k * 19
        b.append(txt(px + 16, yy, lab.split("  ")[0], size=10.5, fill=T["mut"]))
        b.append(txt(px + 184, yy, lab.split("  ")[1], size=10.5, fill=T["acc"], w=600, anchor="end", mono=True))
    # DataFrame
    dx = 570
    b.append(rect(dx, cy - 40, 150, 118, r=12, fill=T["card"], stroke=T["rule"]))
    b.append(txt(dx + 75, cy - 18, "DataFrame", size=12.5, fill=T["ink"], w=600, anchor="middle"))
    b.append(rect(dx + 16, cy - 4, 118, 62, r=6, fill=T["soft"]))
    for k in range(3):
        b.append(line(dx + 16, cy + 16 + k * 18, dx + 134, cy + 16 + k * 18, stroke=T["hair"]))
    b.append(line(dx + 62, cy - 4, dx + 62, cy + 58, stroke=T["hair"]))
    b.append(arrow(238, cy + 18, 296, color=T["mut"]))
    b.append(arrow(508, cy + 18, 566, color=T["mut"]))
    b.append(txt(40, cy + 108, "Stolperfallen:", size=11, fill=T["warn"], w=700))
    b.append(txt(140, cy + 108, "„;“ statt „,“  ·  Latin-1 statt UTF-8  ·  Dezimalkomma  ·  fehlender Header",
                 size=11, fill=T["mut"]))
    return canvas(760, 250, "".join(b),
                  "Einlese-Pipeline: Aus einer Rohdatei wird über bewusste Parser-Entscheidungen "
                  "(Trennzeichen, Encoding, Dezimalzeichen, Header) ein sauberer DataFrame.")


# --- 06 · Tidy Data: long und wide ------------------------------------------
def tidy_long_wide() -> str:
    b = [eyebrow(40, 40, "Tidy Data · dasselbe, zwei Formen")]
    # WIDE
    wx, wy = 40, 96
    wide_head = ["patient_id", "tag_1", "tag_2", "tag_3"]
    wide_rows = [["1042", "3.4", "2.1", "1.8"], ["1043", "1.6", "1.5", "NA"]]
    cw = 74
    b.append(txt(wx, wy - 14, "wide", size=12, fill=T["ink"], w=600, mono=True))
    b.append(rect(wx, wy, cw * len(wide_head), 30 + 26 * len(wide_rows), r=10, fill=T["card"], stroke=T["rule"]))
    for j, h in enumerate(wide_head):
        b.append(txt(wx + j * cw + 12, wy + 20, h, size=10.5, fill=T["ink"], w=600, mono=True))
    b.append(line(wx + 12, wy + 30, wx + cw * len(wide_head) - 12, wy + 30, stroke=T["hair"]))
    for i, r in enumerate(wide_rows):
        for j, v in enumerate(r):
            b.append(txt(wx + j * cw + 12, wy + 48 + i * 26, v, size=10.5, fill=T["ink2"], mono=True))
    # LONG
    lx, ly = 470, 70
    long_head = ["patient_id", "tag", "laktat"]
    long_rows = [["1042", "tag_1", "3.4"], ["1042", "tag_2", "2.1"], ["1042", "tag_3", "1.8"],
                 ["1043", "tag_1", "1.6"], ["1043", "tag_2", "1.5"]]
    lcw = 82
    b.append(txt(lx, ly - 14, "long", size=12, fill=T["ink"], w=600, mono=True))
    b.append(rect(lx, ly, lcw * len(long_head), 30 + 24 * len(long_rows), r=10, fill=T["card"], stroke=T["rule"]))
    for j, h in enumerate(long_head):
        b.append(txt(lx + j * lcw + 12, ly + 20, h, size=10.5, fill=T["ink"], w=600, mono=True))
    b.append(line(lx + 12, ly + 30, lx + lcw * len(long_head) - 12, ly + 30, stroke=T["hair"]))
    for i, r in enumerate(long_rows):
        for j, v in enumerate(r):
            b.append(txt(lx + j * lcw + 12, ly + 46 + i * 24, v, size=10.5, fill=T["ink2"], mono=True))
    # Transform-Pfeile
    mx1, mx2, myc = wx + cw * len(wide_head), lx, 150
    b.append(arrow(mx1 + 10, myc - 12, mx2 - 10, color=T["acc"]))
    b.append(txt((mx1 + mx2) / 2, myc - 20, "pivot_longer · melt", size=10.5, fill=T["acc"], w=600, anchor="middle", mono=True))
    b.append(f'<path d="M{mx2-10},{myc+12} L{mx1+10},{myc+12}" stroke="{T["mut"]}" stroke-width="1.6" fill="none"/>')
    b.append(f'<path d="M{mx1+10},{myc+12} l7,-4 M{mx1+10},{myc+12} l7,4" stroke="{T["mut"]}" stroke-width="1.6" fill="none" stroke-linecap="round"/>')
    b.append(txt((mx1 + mx2) / 2, myc + 26, "pivot_wider · pivot", size=10.5, fill=T["mut"], anchor="middle", mono=True))
    return canvas(760, 250, "".join(b),
                  "Tidy Data: dieselben Laktatverläufe im wide-Format (eine Spalte pro Tag) und im long-Format "
                  "(eine Zeile pro Messung); pivot_longer/melt und pivot_wider wandeln ineinander um.")


# --- 11 · Eine Quelle, viele Formate ----------------------------------------
def eine_quelle() -> str:
    b = [eyebrow(40, 40, "Eine Quelle · viele Formate")]
    sx, sy = 40, 92
    b.append(rect(sx, sy, 200, 150, r=12, fill=T["card"], stroke=T["rule"]))
    b.append(txt(sx + 16, sy + 26, "analyse.qmd", size=12.5, fill=T["ink"], w=600, mono=True))
    b.append(line(sx + 16, sy + 38, sx + 184, sy + 38, stroke=T["hair"]))
    seg = [("text", T["mut"], 120), ("```code```", T["acc"], 150), ("text", T["mut"], 90),
           ("Ergebnis", T["ok"], 110)]
    yy = sy + 58
    for lab, col, wd in seg:
        b.append(rect(sx + 16, yy - 11, wd, 15, r=4, fill=T["soft"]))
        b.append(txt(sx + 22, yy, lab, size=10, fill=col, mono=("code" in lab or "Erg" in lab)))
        yy += 24
    # render
    b.append(arrow(250, sy + 75, 320, color=T["mut"]))
    b.append(txt(285, sy + 66, "render", size=10.5, fill=T["mut"], anchor="middle"))
    outs = [("HTML", 40), ("PDF", 105), ("Word", 170)]
    ox = 340
    for lab, oy in outs:
        b.append(rect(ox, sy + oy - 24, 150, 44, r=10, fill=T["card"], stroke=T["rule"]))
        b.append(rect(ox + 14, sy + oy - 14, 24, 24, r=5, fill=T["acct"]))
        b.append(txt(ox + 48, sy + oy + 4, lab, size=12.5, fill=T["ink"], w=600))
        b.append(f'<path d="M320,{sy+75} C332,{sy+75} 332,{sy+oy-2} {ox},{sy+oy-2}" stroke="{T["rule"]}" stroke-width="1.4" fill="none"/>')
    b.append(txt(sx, sy + 214, "Ein Lauf: Code, Text und Ergebnis bleiben synchron.", size=11.5, fill=T["mut"]))
    return canvas(760, 322, "".join(b),
                  "Reproduzierbarer Bericht: aus einer einzigen Quelldatei (analyse.qmd, die Code, Text und "
                  "Ergebnis vereint) entstehen per render zugleich HTML, PDF und Word.")


# --- 13 · Split, Validierung, Leakage ---------------------------------------
def split_validierung() -> str:
    b = [eyebrow(40, 40, "Aufteilen · validieren · kein Leak")]
    # Split-Balken
    bx, by, bw, bh = 40, 92, 680, 46
    parts = [("Training", 0.6, T["acct"], T["acc2"]), ("Validierung", 0.2, T["okt"], T["ok"]),
             ("Test", 0.2, T["soft"], T["mut"])]
    x = bx
    for lab, frac, fill, fg in parts:
        w = bw * frac
        b.append(rect(x, by, w - 4, bh, r=8, fill=fill))
        b.append(txt(x + w / 2 - 2, by + 21, lab, size=12, fill=fg, w=600, anchor="middle"))
        b.append(txt(x + w / 2 - 2, by + 37, f"{int(frac*100)} %", size=10.5, fill=fg, anchor="middle"))
        x += w
    b.append(txt(bx, by - 12, "Ein Datensatz, drei disjunkte Teile", size=11, fill=T["mut"]))
    # k-fold-Streifen
    fy = by + bh + 44
    b.append(txt(bx, fy - 12, "5-fache Kreuzvalidierung", size=11.5, fill=T["ink"], w=600))
    folds = 5
    fw = (bw - (folds - 1) * 8) / folds
    for r in range(3):
        ry = fy + r * 24
        for k in range(folds):
            fx = bx + k * (fw + 8)
            val = k == r
            b.append(rect(fx, ry, fw, 18, r=4,
                          fill=T["acct"] if val else T["soft"]))
            if val:
                b.append(txt(fx + fw / 2, ry + 13, "Val", size=9.5, fill=T["acc2"], w=600, anchor="middle"))
    b.append(txt(bx + bw + 0, fy + 6, "", size=10))
    # Leakage-Hinweis
    ly = fy + 3 * 24 + 30
    b.append(rect(bx, ly, bw, 40, r=10, fill=T["warnt"]))
    b.append(rect(bx, ly, 3, 40, r=0, fill=T["warn"]))
    b.append(txt(bx + 18, ly + 17, "Leakage vermeiden", size=11.5, fill=T["warn"], w=700))
    b.append(txt(bx + 18, ly + 32, "Skalierung, Imputation und Selektion nur auf dem Training fitten, nie vor dem Split.",
                 size=11, fill=T["ink2"]))
    return canvas(760, ly + 60, "".join(b),
                  "Datenaufteilung in Training, Validierung und Test; 5-fache Kreuzvalidierung rotiert den "
                  "Validierungsblock; Vorverarbeitung wird nur auf dem Training gefittet, um Data Leakage zu vermeiden.")


# --- 02 · Werkzeuge Workspace Setup -----------------------------------------
def workspace_setup() -> str:
    b = [eyebrow(40, 40, "Workspace-Setup · Isolation & Reproduzierbarkeit")]
    
    # System Python (Left)
    b.append(rect(40, 90, 180, 110, r=12, fill=T["soft"]))
    b.append(txt(130, 125, "System Python", size=13, fill=T["ink"], w=600, anchor="middle"))
    b.append(txt(130, 150, "Global / Standard", size=11, fill=T["mut"], anchor="middle"))
    b.append(txt(130, 175, "/usr/bin/python", size=10, fill=T["faint"], anchor="middle", mono=True))
    
    # Project Workspace (Right)
    b.append(rect(300, 90, 420, 200, r=12, fill=T["card"], stroke=T["rule"]))
    b.append(txt(320, 122, "Projektordner (Workspace)", size=14, fill=T["ink"], w=700))
    
    # Inside Workspace: .venv
    b.append(rect(320, 140, 180, 130, r=8, fill=T["acct"], stroke=T["acc"], sw=1.2))
    b.append(txt(410, 170, "Virtuelle Umgebung", size=12.5, fill=T["acc2"], w=600, anchor="middle"))
    b.append(txt(410, 195, ".venv / R-libs", size=12, fill=T["acc"], w=700, anchor="middle", mono=True))
    b.append(txt(410, 225, "pandas, scipy, gt...", size=10.5, fill=T["mut"], anchor="middle"))
    
    # Inside Workspace: Code/Data
    b.append(rect(520, 140, 180, 130, r=8, fill=T["soft"]))
    b.append(txt(610, 170, "Projektcode & Daten", size=12.5, fill=T["ink2"], w=600, anchor="middle"))
    b.append(txt(610, 195, "module/ , data/", size=11.5, fill=T["mut"], anchor="middle", mono=True))
    b.append(txt(610, 225, "Absolute Trennung", size=11, fill=T["ok"], w=600, anchor="middle"))
    
    # Arrow between System Python and .venv: Crossed out!
    b.append(line(220, 145, 292, 145, stroke=T["death"], sw=1.6))
    b.append(f'<path d="M292,145 l-7,-4 M292,145 l-7,4" stroke="{T["death"]}" stroke-width="1.6" fill="none"/>')
    # Cross out symbol
    b.append(line(246, 135, 266, 155, stroke=T["death"], sw=2))
    b.append(txt(256, 125, "keine globale Installation", size=9.5, fill=T["death"], w=600, anchor="middle"))
    
    return canvas(760, 320, "".join(b), "Workspace Setup isolation diagram")


# --- 12 · Missingness Mechanismen -------------------------------------------
def missing_mechanismen() -> str:
    b = [eyebrow(40, 40, "Missingness-Mechanismen · MCAR vs. MAR vs. MNAR")]
    cw, ch = 200, 180
    
    def draw_panel(x, title, subtitle, desc, missing_logic):
        parts = [rect(x, 80, cw, ch, r=10, fill=T["card"], stroke=T["rule"]),
                 txt(x + cw / 2, 105, title, size=13, fill=T["ink"], w=700, anchor="middle"),
                 txt(x + cw / 2, 122, subtitle, size=10, fill=T["mut"], anchor="middle")]
        
        # Draw 5 data rows
        y_start = 140
        for i in range(5):
            ry = y_start + i * 20
            is_missing = missing_logic(i)
            # Dot representing patients
            parts.append(rect(x + 20, ry, 60, 14, r=4, fill=T["soft"]))
            parts.append(txt(x + 25, ry + 11, f"Pat. {i+1}", size=9, fill=T["ink2"], mono=True))
            
            if is_missing:
                parts.append(rect(x + 90, ry, 90, 14, r=4, fill=T["deatht"]))
                parts.append(txt(x + 135, ry + 11, "NA (Fehlt)", size=9, fill=T["death"], w=700, anchor="middle"))
            else:
                parts.append(rect(x + 90, ry, 90, 14, r=4, fill=T["acct"]))
                parts.append(txt(x + 135, ry + 11, "Wert gemessen", size=9, fill=T["acc2"], anchor="middle"))
        
        parts.append(txt(x + cw / 2, 280, desc, size=10, fill=T["mut"], anchor="middle"))
        return "".join(parts)
        
    b.append(draw_panel(40, "MCAR", "Völlig zufällig (Random)", "Zufälliger Laborfehler", lambda i: i in (1, 3)))
    b.append(draw_panel(280, "MAR", "Systematisch (Beobachtet)", "Ältere werden eher gemessen", lambda i: i in (0, 1)))
    b.append(draw_panel(520, "MNAR", "Systematisch (Unbeobachtet)", "Sehr Kranke sterben vor Messung", lambda i: i in (3, 4)))
    
    return canvas(760, 310, "".join(b), "Missingness mechanisms comparison MCAR MAR MNAR")


# --- 13 · Kausale Inferenz DAG ----------------------------------------------
def kausale_dag() -> str:
    import numpy as np
    b = [eyebrow(40, 40, "Kausale Inferenz · DAG & Confounding")]
    
    # Helper to draw node circle with text
    def node(x, y, label, col, bg):
        return (f'<circle cx="{x}" cy="{y}" r="26" fill="{bg}" stroke="{col}" stroke-width="1.8"/>'
                + txt(x, y + 4, label, size=14, fill=col, w=700, anchor="middle", mono=True))
                
    b.append(node(150, 180, "X", T["acc"], T["acct"]))
    b.append(node(550, 180, "Y", T["ok"], T["okt"]))
    b.append(node(350, 90, "Z", T["warn"], T["warnt"]))
    b.append(node(350, 270, "C", T["death"], T["deatht"]))
    
    def arrow_line(x1, y1, x2, y2, color):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        sx1, sy1 = x1 + ux * 28, y1 + uy * 28
        sx2, sy2 = x2 - ux * 30, y2 - uy * 30
        
        ax, ay = sx2 - ux * 8, sy2 - uy * 8
        px, py = -uy * 5, ux * 5
        head = f'M{sx2},{sy2} L{ax+px},{ay+py} L{ax-px},{ay-py} Z'
        return (line(sx1, sy1, sx2, sy2, stroke=color, sw=2)
                + f'<path d="{head}" fill="{color}"/>')
                
    b.append(arrow_line(150, 180, 550, 180, T["acc"]))
    b.append(arrow_line(350, 90, 150, 180, T["warn"]))
    b.append(arrow_line(350, 90, 550, 180, T["warn"]))
    b.append(arrow_line(150, 180, 350, 270, T["mut"]))
    b.append(arrow_line(550, 180, 350, 270, T["mut"]))
    
    b.append(txt(150, 222, "Intervention (X)", size=11, fill=T["acc2"], w=600, anchor="middle"))
    b.append(txt(550, 222, "Outcome (Y)", size=11, fill=T["ok"], w=600, anchor="middle"))
    b.append(txt(350, 48, "Confounder (Z)", size=11, fill=T["warn"], w=700, anchor="middle"))
    b.append(txt(350, 312, "Collider (C)", size=11, fill=T["death"], w=700, anchor="middle"))
    
    b.append(rect(40, 240, 96, 50, r=6, fill=T["warnt"]))
    b.append(txt(45, 258, "Z adjustieren!", size=10, fill=T["warn"], w=700))
    b.append(txt(45, 276, "(blockt Backdoor)", size=9, fill=T["mut"]))
    
    b.append(rect(610, 240, 110, 50, r=6, fill=T["deatht"]))
    b.append(txt(615, 258, "C NICHT adjustieren!", size=10, fill=T["death"], w=700))
    b.append(txt(615, 276, "(erzeugt Scheinkorr.)", size=9, fill=T["mut"]))
    
    return canvas(760, 330, "".join(b), "Causal DAG representation showing Confounder and Collider adjustment rules")


# --- Erzeugen ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
TARGETS = {
    "02-werkzeuge": [("workspace_setup", workspace_setup)],
    "05-sql": [("datenmodell", datenmodell), ("join_typen", join_typen)],
    "03-grundlagen": [("dataframe_anatomie", dataframe_anatomie)],
    "04-daten-einlesen": [("einlesen_pipeline", einlesen_pipeline)],
    "06-transformation": [("tidy_long_wide", tidy_long_wide)],
    "14-fehlende-werte": [("missing_mechanismen", missing_mechanismen)],
    "15-kausale-inferenz": [("kausale_dag", kausale_dag)],
    "22-reproduzierbare-berichte": [("eine_quelle", eine_quelle)],
    "24-praediktion-workflow": [("split_validierung", split_validierung)],
}


def main() -> None:
    n = 0
    for slug, diagrams in TARGETS.items():
        adir = ROOT / "module" / slug / "assets"
        adir.mkdir(parents=True, exist_ok=True)
        for name, fn in diagrams:
            (adir / f"{name}.svg").write_text(fn(), encoding="utf-8")
            n += 1
    print(f"{n} Diagramme erzeugt in module/*/assets/")


if __name__ == "__main__":
    main()
