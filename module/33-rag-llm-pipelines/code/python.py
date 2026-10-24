"""Module 33 - a small, offline-runnable RAG *retrieval* pipeline: chunking, a
vector-store lookup (TF-IDF + cosine similarity via scikit-learn's
NearestNeighbors), a top-k retriever, a labelled retrieval evaluation
(precision@k / recall@k), and an "answer not in context" guard whose
threshold is *calibrated* on a small labelled set rather than hand-picked.
The guard's precision/recall/F1 are then evaluated out-of-sample via
leave-one-out cross-validation (see evaluate_guard_loocv) -- reporting them
on the same data used to pick the threshold would be circular.

There is deliberately NO generation step here: the module demonstrates
retrieval and grounding (finding and returning the relevant passage), not the
LLM answer-writing that would sit on top of it. So it cannot and does not claim
to "prevent hallucinations" – only a grounded generator could, and evaluating
that is out of scope.

Run: python module/33-rag-llm-pipelines/code/python.py
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

TOP_K = 3  # retriever returns the top-3 chunks (loesungen.md inspects "Top 3")

# A short synthetic sepsis guideline excerpt (six recommendations). Deliberately
# does NOT state an exact antibiotic dose, so the out-of-context questions below
# have no correct answer in the corpus -- this is what the guard must catch.
GUIDELINE = """Bei Sepsis-induzierter Hypotension oder einem Laktatwert von mindestens 4 mmol/L sollen innerhalb der ersten 3 Stunden mindestens 30 ml/kg Koerpergewicht einer balancierten kristalloiden Infusionsloesung verabreicht werden. Der Effekt der Volumengabe soll ueber dynamische Parameter wie die Pulsdruckvariation oder ein passives Beinheben ueberprueft werden, um eine Fluessigkeitsueberladung zu vermeiden.

Als Vasopressor der ersten Wahl bei septischem Schock soll Noradrenalin eingesetzt werden, mit dem Ziel, einen mittleren arteriellen Druck (MAP) von mindestens 65 mmHg zu erreichen. Reicht Noradrenalin allein nicht aus, kann Vasopressin als zweiter Vasopressor ergaenzt werden, um die Noradrenalin-Dosis zu senken. Adrenalin gilt als Reservemedikament bei therapierefraktaerem Schock.

Eine kalkulierte intravenoese Antibiotikatherapie soll so frueh wie moeglich, spaetestens jedoch innerhalb einer Stunde nach Diagnosestellung einer Sepsis begonnen werden. Vor Gabe der Antibiotika sollen, sofern ohne relevante Verzoegerung moeglich, mindestens zwei Blutkulturpaare abgenommen werden. Die Auswahl des Antibiotikums richtet sich nach dem vermuteten Infektionsfokus und der lokalen Resistenzlage.

Der Infektionsfokus soll innerhalb der ersten 6 bis 12 Stunden identifiziert und, wo moeglich, chirurgisch oder interventionell saniert werden (Fokuskontrolle), zum Beispiel durch Drainage eines Abszesses oder Entfernung eines infizierten Katheters. Eine verzoegerte Fokuskontrolle ist mit einer hoeheren Sterblichkeit assoziiert.

Bei Patient:innen mit akutem Nierenversagen und haemodynamischer Instabilitaet soll eine kontinuierliche Nierenersatztherapie (CRRT) gegenueber intermittierenden Verfahren bevorzugt werden, da sie den Kreislauf schonender entlastet. Die Indikation richtet sich nach Diurese, Elektrolytentgleisung und Fluessigkeitsbilanz, nicht allein nach dem Kreatininwert.

Der Therapieerfolg soll durch wiederholte Laktatmessungen ueberwacht werden. Ein Absinken des Laktatwertes um mindestens 20 Prozent innerhalb von 2 Stunden gilt als Hinweis auf ein Ansprechen der Therapie. Ein persistierend erhoehtes Laktat trotz adaequater Volumen- und Vasopressortherapie ist ein Warnzeichen fuer eine unzureichende Fokuskontrolle."""

# Labelled evaluation set (a tiny "gold" set: n=8, 5 in-context / 3
# out-of-context -- too small for a meaningful calibration/held-out split).
# Each in-context question names a distinctive phrase from the guideline
# chunk that actually answers it, so we can locate the relevant chunk
# automatically; out-of-context questions have no answer in the corpus. This
# set is used for BOTH the retrieval metrics (precision@k / recall@k) AND to
# calibrate + evaluate the guard threshold (the latter via leave-one-out CV,
# see evaluate_guard_loocv, so the reported guard metrics are out-of-sample).
EVAL = [
    {"frage": "Welcher Vasopressor wird bei Sepsis empfohlen?",
     "beleg": "Noradrenalin", "im_kontext": True},
    {"frage": "Was steht zur Nierenersatztherapie?",
     "beleg": "Nierenersatztherapie", "im_kontext": True},
    {"frage": "Wie schnell muss die Antibiotikatherapie beginnen?",
     "beleg": "innerhalb einer Stunde", "im_kontext": True},
    {"frage": "Woran erkennt man ein Ansprechen der Therapie?",
     "beleg": "Absinken des Laktatwertes", "im_kontext": True},
    {"frage": "Wie viel Volumen soll in den ersten 3 Stunden gegeben werden?",
     "beleg": "30 ml/kg", "im_kontext": True},
    {"frage": "Welche exakte Antibiotikadosis ist genannt?",
     "beleg": None, "im_kontext": False},
    {"frage": "Welche Beatmungsstrategie wird empfohlen?",
     "beleg": None, "im_kontext": False},
    {"frage": "Welcher Blutzucker-Zielwert gilt bei Sepsis?",
     "beleg": None, "im_kontext": False},
]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split a document into overlapping chunks for indexing.

    Splits on paragraph breaks first (keeps a recommendation intact whenever it
    fits in one chunk); only a paragraph longer than `chunk_size` is further
    cut with a sliding window of `overlap` characters, so no sentence is lost
    at a chunk boundary purely by bad luck.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            end = start + chunk_size
            piece = para[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(para):
                break
            start = end - overlap
    return chunks


def build_vector_store(chunks: list[str]) -> tuple[TfidfVectorizer, NearestNeighbors]:
    """Embed chunks with TF-IDF and index them for nearest-neighbor lookup.

    This NearestNeighbors index *is* a minimal vector store: it holds one
    vector per chunk and answers "which vectors are closest to this query
    vector?". A production system would swap this in-memory scikit-learn index
    for a persistent vector database such as ChromaDB or FAISS -- same
    question, but scaled to millions of chunks and saved to disk. The retrieval
    math (TF-IDF + cosine similarity) taught here stays identical.
    """
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    k = min(TOP_K, len(chunks))
    store = NearestNeighbors(n_neighbors=k, metric="cosine").fit(chunk_vectors)
    return vectorizer, store


def retrieve(question: str, vectorizer: TfidfVectorizer, store: NearestNeighbors,
             chunks: list[str], k: int = TOP_K) -> list[tuple[int, str, float]]:
    """Return the top-k matching chunks as (index, text, cosine similarity),
    ordered best first."""
    question_vector = vectorizer.transform([question])
    n = min(k, len(chunks))
    distances, indices = store.kneighbors(question_vector, n_neighbors=n)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append((int(idx), chunks[int(idx)], 1.0 - float(dist)))
    return results


def gold_chunk_index(beleg: str, chunks: list[str]) -> int:
    """Index of the chunk that contains the gold evidence phrase."""
    for i, c in enumerate(chunks):
        if beleg in c:
            return i
    raise ValueError(f"Evidence phrase not found in any chunk: {beleg!r}")


def evaluate_retrieval(chunks: list[str], vectorizer, store, k: int = TOP_K) -> None:
    """Precision@k and recall@k over the in-context gold questions.

    Each in-context question has exactly one relevant chunk, so recall@k is the
    hit rate (was the gold chunk among the top-k?) and precision@k is that hit
    divided by k, averaged over questions.
    """
    in_context = [e for e in EVAL if e["im_kontext"]]
    recalls, precisions = [], []
    print(f"\n=== Retrieval-Evaluation (Gold-Set, k={k}) ===")
    for e in in_context:
        gold = gold_chunk_index(e["beleg"], chunks)
        top = retrieve(e["frage"], vectorizer, store, chunks, k=k)
        top_idx = [idx for idx, _, _ in top]
        hit = gold in top_idx
        recalls.append(1.0 if hit else 0.0)
        precisions.append((1.0 / k) if hit else 0.0)
        rank = top_idx.index(gold) + 1 if hit else None
        status = f"auf Rang {rank}" if hit else "nicht in Top-k"
        print(f"  {'OK  ' if hit else 'MISS'} Gold-Chunk {gold} {status}: {e['frage']}")
    print(f"  recall@{k}:    {np.mean(recalls):.3f}   "
          f"(Anteil Fragen, deren Gold-Chunk in den Top-{k} liegt)")
    print(f"  precision@{k}: {np.mean(precisions):.3f}   "
          f"(relevante Treffer / {k}, gemittelt)")


def guard_scores_labels(chunks: list[str], vectorizer, store) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 cosine similarity and in-context label for every EVAL question."""
    scores, labels = [], []
    for e in EVAL:
        top = retrieve(e["frage"], vectorizer, store, chunks, k=1)
        scores.append(top[0][2])
        labels.append(1 if e["im_kontext"] else 0)
    return np.asarray(scores), np.asarray(labels)


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    """F1-maximising threshold over candidate midpoints between observed scores.

    Returns (threshold, precision, recall, f1) computed on the (scores,
    labels) passed in. The caller decides whether that's the full labelled
    set (an in-sample fit, fine for picking a deployment threshold) or a
    leave-one-out training fold (used for out-of-sample evaluation below).
    """
    order = np.sort(np.unique(scores))
    candidates = [(order[i] + order[i + 1]) / 2 for i in range(len(order) - 1)]
    candidates += [order[0] - 1e-6, order[-1] + 1e-6]

    best_t, best_f1, best_pr, best_rc = candidates[0], -1.0, 0.0, 0.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_t, best_f1, best_pr, best_rc = t, f1, precision, recall
    return float(best_t), best_pr, best_rc, best_f1


def calibrate_threshold(chunks: list[str], vectorizer, store) -> float:
    """Pick the *deployed* guard threshold on the full labelled set.

    This is the threshold section 3 below actually applies to new questions.
    It is fit on all n=8 labelled questions, so it is an in-sample choice --
    a normal thing to do when picking a hyperparameter for deployment. What
    would NOT be normal is reporting the precision/recall/F1 that comes out
    of evaluating this same fit on the same n=8 questions as "the" guard
    performance -- that number is circular by construction (it will always
    look at least as good as it can be made to look, because the threshold
    was chosen to maximise exactly that score on exactly that data). The
    honest, out-of-sample metrics are computed separately in
    evaluate_guard_loocv().
    """
    scores, labels = guard_scores_labels(chunks, vectorizer, store)
    t, _pr, _rc, _f1 = best_threshold(scores, labels)
    print("\n=== Schwelle fuer den Guard (auf allen n=8 gelabelten Fragen gewaehlt) ===")
    print(f"  In-Kontext-Top-1-Scores:      "
          f"{sorted(np.round(scores[labels == 1], 3).tolist(), reverse=True)}")
    print(f"  Ausser-Kontext-Top-1-Scores:  "
          f"{sorted(np.round(scores[labels == 0], 3).tolist(), reverse=True)}")
    print(f"  gewaehlte Schwelle (Deployment): {t:.3f}")
    print("  Diese Schwelle wird unten auf Beispielfragen angewandt. Ihre In-Sample-Guete auf")
    print("  denselben n=8 Fragen waere zirkulaer und wird deshalb NICHT berichtet -- die")
    print("  ehrlichen Out-of-Sample-Kennzahlen liefert die Leave-one-out-Auswertung unten.")
    return t


def evaluate_guard_loocv(chunks: list[str], vectorizer, store) -> None:
    """Out-of-sample precision/recall/F1 for the guard via leave-one-out CV.

    Protocol (stated explicitly because it matters): the labelled set has
    only n=8 questions (5 in-context, 3 out-of-context) -- too small to split
    into a meaningful calibration half and evaluation half. Instead, for each
    question we pick the F1-maximising threshold on the OTHER n-1 questions
    (calibrate_threshold's best_threshold(), just on a leave-one-out fold)
    and predict the held-out question with that threshold. Every prediction
    therefore comes from a threshold that never saw that question's own
    label, which is what makes the pooled precision/recall/F1 below
    out-of-sample rather than circular.
    """
    scores, labels = guard_scores_labels(chunks, vectorizer, store)
    n = len(scores)
    preds = np.zeros(n, dtype=int)
    print(f"\n=== Guard-Evaluation: Leave-one-out ueber n={n} gelabelte Fragen ===")
    print("  Protokoll: pro Frage die Schwelle auf den jeweils anderen n-1 Fragen waehlen,")
    print("  dann NUR auf der zurueckgehaltenen Frage vorhersagen (Out-of-Sample).")
    for i in range(n):
        train_scores = np.delete(scores, i)
        train_labels = np.delete(labels, i)
        t_i, _pr, _rc, _f1 = best_threshold(train_scores, train_labels)
        pred_i = int(scores[i] >= t_i)
        preds[i] = pred_i
        status = "richtig" if pred_i == labels[i] else "FALSCH"
        label_txt = "im Kontext" if labels[i] else "ausserhalb"
        pred_txt = "im Kontext" if pred_i else "ausserhalb"
        print(f"    Frage {i + 1}: Score={scores[i]:.3f}  Label={label_txt}  "
              f"Schwelle(ohne diese Frage)={t_i:.3f}  Vorhersage={pred_txt}  [{status}]")

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\n  Gepoolte Out-of-Sample-Kennzahlen ueber alle {n} Fragen "
          f"(tp={tp}, fp={fp}, fn={fn}):")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    if f1 < 1.0:
        print("  Nicht perfekt -- das ist das ehrliche Ergebnis. Die In-Sample-Zahl")
        print("  (hier Precision=Recall=F1=1.00, auf denselben Daten berechnet, die die Schwelle")
        print("  festgelegt haben) waere zirkulaer und sagt nichts ueber neue Fragen aus.")
    else:
        print("  Auch out-of-sample perfekt auf diesem winzigen Set -- bei n=8 Fragen ist das")
        print("  ein einzelner guenstiger Datenpunkt, kein Beleg fuer Generalisierung.")


def main() -> None:
    chunks = chunk_text(GUIDELINE)
    print(f"Dokument in {len(chunks)} Chunks zerlegt (Zielgroesse 500 Zeichen, 50 Zeichen Ueberlappung).")

    vectorizer, store = build_vector_store(chunks)

    # 1) Does the retriever put the right chunk in the top-k?
    evaluate_retrieval(chunks, vectorizer, store)

    # 2) Calibrate the deployed "not in context" guard threshold on the full
    #    labelled set, then evaluate the guard's actual precision/recall/F1
    #    out-of-sample via leave-one-out CV (2a) -- reporting in-sample
    #    metrics here would be circular, see the docstrings above.
    threshold = calibrate_threshold(chunks, vectorizer, store)
    evaluate_guard_loocv(chunks, vectorizer, store)

    # 3) Apply the retriever + calibrated guard to example questions, showing
    #    the top-k chunks so a reader can inspect them (as loesungen.md asks).
    print("\n=== Retrieval + Guard auf Beispielfragen ===")
    questions = [
        "Welcher Vasopressor wird bei Sepsis empfohlen?",
        "Was steht zur Nierenersatztherapie?",
        "Welche exakte Antibiotikadosis ist genannt?",  # not in the corpus
    ]
    for question in questions:
        top = retrieve(question, vectorizer, store, chunks)
        best_score = top[0][2]
        print(f"\nFrage: {question}")
        print(f"  Top-1-Aehnlichkeit (Cosinus): {best_score:.3f}  (Schwelle: {threshold:.3f})")
        if best_score < threshold:
            print("  -> Information in den bereitgestellten Leitlinien nicht enthalten "
                  "(Score unter Schwelle).")
        else:
            print(f"  -> Top-{len(top)} Chunks (zur Pruefung durch Menschen):")
            for rank, (idx, text, score) in enumerate(top, start=1):
                print(f"     {rank}. [Chunk {idx}, Score {score:.3f}] {text[:90]}...")


if __name__ == "__main__":
    main()
