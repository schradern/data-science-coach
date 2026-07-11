"""Figure for module 33: a schematic RAG-pipeline diagram.

Run: python module/33-rag-llm-pipelines/code/figures.py

Purely illustrative (no data-driven numbers) -- built with matplotlib patches
so it lives alongside the module's own code, in the shared course style
(lib.plotstyle colours), rather than in lib/diagrams.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

STEPS = [
    ("Leitlinien-\ntext", "z. B. S3-Sepsis-\nLeitlinie", SECONDARY),
    ("Chunking", "500 Zeichen,\n50 Zeichen Overlap", PRIMARY),
    ("Embedding", "TF-IDF-Vektor\npro Chunk", PRIMARY),
    ("Vektor-Store", "z. B. ChromaDB /\nFAISS (hier: NN-Index)", PRIMARY),
    ("Retrieval", "Top-1 nach\nCosinus-Aehnlichkeit", EVENT),
    ("Antwort +\nZitat", "nur aus dem\nKontext, sonst Hinweis", "#2C7A54"),
]


def fig_rag_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(0, len(STEPS))
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h = 0.78, 0.55
    y_c = 0.55
    for i, (title, sub, color) in enumerate(STEPS):
        x_c = i + 0.5
        box = FancyBboxPatch((x_c - box_w / 2, y_c - box_h / 2), box_w, box_h,
                              boxstyle="round,pad=0.02,rounding_size=0.06",
                              linewidth=1.4, edgecolor=color, facecolor="white", zorder=3)
        ax.add_patch(box)
        ax.text(x_c, y_c + 0.08, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="#16181C", zorder=4)
        ax.text(x_c, y_c - 0.16, sub, ha="center", va="center", fontsize=8.3,
                color="#6B7178", zorder=4)
        if i < len(STEPS) - 1:
            arrow = FancyArrowPatch((x_c + box_w / 2 + 0.02, y_c), (x_c + 1 - box_w / 2 - 0.02, y_c),
                                     arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
                                     color="#9AA0A8", zorder=2)
            ax.add_patch(arrow)

    ax.text(0.5, 0.06, "Schritte 2-4 laufen einmalig beim Indexieren; Schritt 5-6 bei jeder Frage.",
            ha="left", va="center", fontsize=9, color="#6B7178", transform=ax.transAxes)
    ax.set_title("RAG-Pipeline: von der Leitlinie zur belegten Antwort", loc="left", fontsize=13.5,
                 fontweight="bold", color="#16181C", y=1.02)
    save(fig, ASSETS / "rag_pipeline.png")


def main() -> None:
    apply_style()
    fig_rag_pipeline()


if __name__ == "__main__":
    main()
