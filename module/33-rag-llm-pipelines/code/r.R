# Module 33 - a small, offline-runnable RAG *retrieval* pipeline: chunking, a
# vector-store lookup (TF-IDF + cosine similarity, from scratch in base R -- the
# same documented algorithm as code/python.py), a top-k retriever, a labelled
# retrieval evaluation (precision@k / recall@k), and an "answer not in context"
# guard whose threshold is CALIBRATED on a small labelled set, not hand-picked.
#
# There is deliberately NO generation step: this demonstrates retrieval and
# grounding, not LLM answer-writing, and therefore makes no hallucination claim.
#   Rscript module/33-rag-llm-pipelines/code/r.R

# A short synthetic sepsis guideline excerpt (six recommendations). Deliberately
# does NOT state an exact antibiotic dose, so the third question below has no
# correct answer in the corpus -- this is what the confidence threshold must catch.
GUIDELINE <- paste(
  "Bei Sepsis-induzierter Hypotension oder einem Laktatwert von mindestens 2 mmol/L sollen innerhalb der ersten 3 Stunden mindestens 30 ml/kg Koerpergewicht einer balancierten kristalloiden Infusionsloesung verabreicht werden. Der Effekt der Volumengabe soll ueber dynamische Parameter wie die Pulsdruckvariation oder ein passives Beinheben ueberprueft werden, um eine Fluessigkeitsueberladung zu vermeiden.",
  "Als Vasopressor der ersten Wahl bei septischem Schock soll Noradrenalin eingesetzt werden, mit dem Ziel, einen mittleren arteriellen Druck (MAP) von mindestens 65 mmHg zu erreichen. Reicht Noradrenalin allein nicht aus, kann Vasopressin als zweiter Vasopressor ergaenzt werden, um die Noradrenalin-Dosis zu senken. Adrenalin gilt als Reservemedikament bei therapierefraktaerem Schock.",
  "Eine kalkulierte intravenoese Antibiotikatherapie soll so frueh wie moeglich, spaetestens jedoch innerhalb einer Stunde nach Diagnosestellung einer Sepsis begonnen werden. Vor Gabe der Antibiotika sollen, sofern ohne relevante Verzoegerung moeglich, mindestens zwei Blutkulturpaare abgenommen werden. Die Auswahl des Antibiotikums richtet sich nach dem vermuteten Infektionsfokus und der lokalen Resistenzlage.",
  "Der Infektionsfokus soll innerhalb der ersten 6 bis 12 Stunden identifiziert und, wo moeglich, chirurgisch oder interventionell saniert werden (Fokuskontrolle), zum Beispiel durch Drainage eines Abszesses oder Entfernung eines infizierten Katheters. Eine verzoegerte Fokuskontrolle ist mit einer hoeheren Sterblichkeit assoziiert.",
  "Bei Patient:innen mit akutem Nierenversagen und haemodynamischer Instabilitaet soll eine kontinuierliche Nierenersatztherapie (CRRT) gegenueber intermittierenden Verfahren bevorzugt werden, da sie den Kreislauf schonender entlastet. Die Indikation richtet sich nach Diurese, Elektrolytentgleisung und Fluessigkeitsbilanz, nicht allein nach dem Kreatininwert.",
  "Der Therapieerfolg soll durch wiederholte Laktatmessungen ueberwacht werden. Ein Absinken des Laktatwertes um mindestens 20 Prozent innerhalb von 2 Stunden gilt als Hinweis auf ein Ansprechen der Therapie. Ein persistierend erhoehtes Laktat trotz adaequater Volumen- und Vasopressortherapie ist ein Warnzeichen fuer eine unzureichende Fokuskontrolle.",
  sep = "\n\n"
)

TOP_K <- 3  # retriever returns the top-3 chunks (loesungen.md inspects "Top 3")

# Labelled evaluation set (tiny "gold" set: n=8, 5 in-context / 3
# out-of-context -- too small for a meaningful calibration/held-out split),
# same as code/python.py. Each in-context question names a distinctive
# phrase from the chunk that answers it; out-of-context questions have no
# answer in the corpus. Used for BOTH the retrieval metrics AND to calibrate
# + evaluate the guard threshold (the latter via leave-one-out CV below, so
# the reported guard metrics are out-of-sample, not circular).
EVAL <- list(
  list(frage = "Welcher Vasopressor wird bei Sepsis empfohlen?",            beleg = "Noradrenalin",              im_kontext = TRUE),
  list(frage = "Was steht zur Nierenersatztherapie?",                       beleg = "Nierenersatztherapie",       im_kontext = TRUE),
  list(frage = "Wie schnell muss die Antibiotikatherapie beginnen?",        beleg = "innerhalb einer Stunde",     im_kontext = TRUE),
  list(frage = "Woran erkennt man ein Ansprechen der Therapie?",            beleg = "Absinken des Laktatwertes",  im_kontext = TRUE),
  list(frage = "Wie viel Volumen soll in den ersten 3 Stunden gegeben werden?", beleg = "30 ml/kg",              im_kontext = TRUE),
  list(frage = "Welche exakte Antibiotikadosis ist genannt?",               beleg = NA,                           im_kontext = FALSE),
  list(frage = "Welche Beatmungsstrategie wird empfohlen?",                 beleg = NA,                           im_kontext = FALSE),
  list(frage = "Welcher Blutzucker-Zielwert gilt bei Sepsis?",              beleg = NA,                           im_kontext = FALSE)
)

chunk_text <- function(text, chunk_size = 500, overlap = 50) {
  paragraphs <- Filter(function(p) nchar(p) > 0, trimws(strsplit(text, "\n\n")[[1]]))
  chunks <- character(0)
  for (para in paragraphs) {
    if (nchar(para) <= chunk_size) {
      chunks <- c(chunks, para)
      next
    }
    start <- 1
    n <- nchar(para)
    repeat {
      end <- min(start + chunk_size - 1, n)
      chunks <- c(chunks, trimws(substr(para, start, end)))
      if (end >= n) break
      start <- end - overlap + 1
    }
  }
  chunks
}

tokenize <- function(text) {
  text <- tolower(text)
  text <- gsub("[^a-z0-9äöüß ]", " ", text)
  tokens <- strsplit(text, "\\s+")[[1]]
  tokens[tokens != ""]
}

# TF-IDF with smoothed IDF and L2-normalized rows -- the same formula
# scikit-learn's TfidfVectorizer uses by default, reimplemented here in base R
# so the R track needs no extra packages to run offline.
build_tfidf <- function(chunks) {
  docs_tokens <- lapply(chunks, tokenize)
  vocab <- sort(unique(unlist(docs_tokens)))
  n_docs <- length(chunks)
  tf <- matrix(0, nrow = n_docs, ncol = length(vocab), dimnames = list(NULL, vocab))
  for (i in seq_along(docs_tokens)) {
    tok <- docs_tokens[[i]]
    counts <- table(tok)
    tf[i, names(counts)] <- as.numeric(counts) / length(tok)
  }
  doc_freq <- colSums(tf > 0)
  idf <- log((n_docs + 1) / (doc_freq + 1)) + 1
  tfidf <- sweep(tf, 2, idf, `*`)
  norms <- sqrt(rowSums(tfidf^2))
  norms[norms == 0] <- 1
  tfidf <- tfidf / norms
  list(matrix = tfidf, vocab = vocab, idf = idf)
}

vectorize_query <- function(text, vocab, idf) {
  tok <- tokenize(text)
  tf <- stats::setNames(rep(0, length(vocab)), vocab)
  counts <- table(tok)
  common <- intersect(names(counts), vocab)
  tf[common] <- as.numeric(counts[common]) / length(tok)
  vec <- tf * idf
  norm <- sqrt(sum(vec^2))
  if (norm > 0) vec <- vec / norm
  vec
}

# The TF-IDF matrix (rows already L2-normalized) IS the "vector store": one
# row per chunk. retrieve() answers "which row is closest (cosine) to the
# query vector?" -- a production system would replace this in-memory matrix
# with a persistent vector database (ChromaDB, FAISS), same lookup at scale.
# Return the top-k chunk indices and cosine similarities, best first.
retrieve <- function(question, index, chunks, k = TOP_K) {
  qvec <- vectorize_query(question, index$vocab, index$idf)
  sims <- apply(index$matrix, 1, function(row) sum(row * qvec))
  ord  <- order(sims, decreasing = TRUE)[seq_len(min(k, length(sims)))]
  list(idx = ord, score = sims[ord])
}

gold_chunk_index <- function(beleg, chunks) {
  which(vapply(chunks, function(c) grepl(beleg, c, fixed = TRUE), logical(1)))[1]
}

chunks <- chunk_text(GUIDELINE)
cat(sprintf("Dokument in %d Chunks zerlegt (Zielgroesse 500 Zeichen, 50 Zeichen Ueberlappung).\n", length(chunks)))

index <- build_tfidf(chunks)

# --- 1) Retrieval evaluation: precision@k / recall@k on the gold set ---------
in_context <- Filter(function(e) e$im_kontext, EVAL)
recalls <- numeric(0); precisions <- numeric(0)
cat(sprintf("\n=== Retrieval-Evaluation (Gold-Set, k=%d) ===\n", TOP_K))
for (e in in_context) {
  gold <- gold_chunk_index(e$beleg, chunks)
  top  <- retrieve(e$frage, index, chunks)
  hit  <- gold %in% top$idx
  recalls    <- c(recalls, if (hit) 1 else 0)
  precisions <- c(precisions, if (hit) 1 / TOP_K else 0)
  rank <- if (hit) which(top$idx == gold) else NA
  status <- if (hit) sprintf("auf Rang %d", rank) else "nicht in Top-k"
  cat(sprintf("  %s Gold-Chunk %d %s: %s\n", if (hit) "OK  " else "MISS", gold, status, e$frage))
}
cat(sprintf("  recall@%d:    %.3f   (Anteil Fragen, deren Gold-Chunk in den Top-%d liegt)\n",
            TOP_K, mean(recalls), TOP_K))
cat(sprintf("  precision@%d: %.3f   (relevante Treffer / %d, gemittelt)\n",
            TOP_K, mean(precisions), TOP_K))

# best_threshold(): F1-maximising threshold over candidate midpoints between
# observed scores. Returns the threshold plus precision/recall/F1 computed on
# the (scores, labels) passed in -- the caller decides whether that's the
# full labelled set (an in-sample fit, fine for a deployment threshold) or a
# leave-one-out training fold (used for out-of-sample evaluation below).
best_threshold <- function(sc, lb) {
  ord <- sort(unique(sc))
  cands <- if (length(ord) > 1) (head(ord, -1) + tail(ord, -1)) / 2 else numeric(0)
  cands <- c(cands, min(ord) - 1e-6, max(ord) + 1e-6)
  best <- list(t = cands[1], f1 = -1, pr = 0, rc = 0)
  for (t in cands) {
    pred <- as.integer(sc >= t)
    tp <- sum(pred == 1 & lb == 1); fp <- sum(pred == 1 & lb == 0); fn <- sum(pred == 0 & lb == 1)
    pr <- if (tp + fp > 0) tp / (tp + fp) else 0
    rc <- if (tp + fn > 0) tp / (tp + fn) else 0
    f1 <- if (pr + rc > 0) 2 * pr * rc / (pr + rc) else 0
    if (f1 > best$f1) best <- list(t = t, f1 = f1, pr = pr, rc = rc)
  }
  best
}

# --- 2) Calibrate the *deployed* guard threshold on the full labelled set ----
# This is the threshold section 3 below actually applies. It is fit on all
# n=8 labelled questions -- fine as a deployment choice, but its in-sample
# precision/recall/F1 would be circular (the threshold was chosen to
# maximise exactly that score on exactly that data), so it is NOT reported
# here. The honest, out-of-sample metrics come from the leave-one-out
# evaluation in step 2a below.
scores <- vapply(EVAL, function(e) retrieve(e$frage, index, chunks, k = 1)$score[1], numeric(1))
labels <- vapply(EVAL, function(e) if (e$im_kontext) 1 else 0, numeric(1))
deploy <- best_threshold(scores, labels)
threshold <- deploy$t
cat("\n=== Schwelle fuer den Guard (auf allen n=8 gelabelten Fragen gewaehlt) ===\n")
cat(sprintf("  In-Kontext-Top-1-Scores:      %s\n", paste(sprintf("%.3f", sort(scores[labels == 1], decreasing = TRUE)), collapse = ", ")))
cat(sprintf("  Ausser-Kontext-Top-1-Scores:  %s\n", paste(sprintf("%.3f", sort(scores[labels == 0], decreasing = TRUE)), collapse = ", ")))
cat(sprintf("  gewaehlte Schwelle (Deployment): %.3f\n", threshold))
cat("  Diese Schwelle wird unten auf Beispielfragen angewandt. Ihre In-Sample-Guete auf\n")
cat("  denselben n=8 Fragen waere zirkulaer und wird deshalb NICHT berichtet -- die\n")
cat("  ehrlichen Out-of-Sample-Kennzahlen liefert die Leave-one-out-Auswertung unten.\n")

# --- 2a) Guard evaluation: leave-one-out CV (out-of-sample precision/recall/F1) --
# Protocol: the labelled set has only n=8 questions (5 in-context, 3
# out-of-context) -- too small to split into a meaningful calibration half
# and evaluation half. Instead, for each question, pick the F1-maximising
# threshold on the OTHER n-1 questions and predict the held-out question
# with that threshold; pool all n held-out predictions into one
# precision/recall/F1. Every prediction comes from a threshold that never
# saw that question's own label, which is what makes this out-of-sample.
n_eval <- length(scores)
preds <- integer(n_eval)
cat(sprintf("\n=== Guard-Evaluation: Leave-one-out ueber n=%d gelabelte Fragen ===\n", n_eval))
cat("  Protokoll: pro Frage die Schwelle auf den jeweils anderen n-1 Fragen waehlen,\n")
cat("  dann NUR auf der zurueckgehaltenen Frage vorhersagen (Out-of-Sample).\n")
for (i in seq_len(n_eval)) {
  fold <- best_threshold(scores[-i], labels[-i])
  pred_i <- as.integer(scores[i] >= fold$t)
  preds[i] <- pred_i
  status <- if (pred_i == labels[i]) "richtig" else "FALSCH"
  label_txt <- if (labels[i] == 1) "im Kontext" else "ausserhalb"
  pred_txt <- if (pred_i == 1) "im Kontext" else "ausserhalb"
  cat(sprintf("    Frage %d: Score=%.3f  Label=%s  Schwelle(ohne diese Frage)=%.3f  Vorhersage=%s  [%s]\n",
              i, scores[i], label_txt, fold$t, pred_txt, status))
}
tp <- sum(preds == 1 & labels == 1); fp <- sum(preds == 1 & labels == 0); fn <- sum(preds == 0 & labels == 1)
pr <- if (tp + fp > 0) tp / (tp + fp) else 0
rc <- if (tp + fn > 0) tp / (tp + fn) else 0
f1 <- if (pr + rc > 0) 2 * pr * rc / (pr + rc) else 0
cat(sprintf("\n  Gepoolte Out-of-Sample-Kennzahlen ueber alle %d Fragen (tp=%d, fp=%d, fn=%d):\n", n_eval, tp, fp, fn))
cat(sprintf("  Precision=%.3f  Recall=%.3f  F1=%.3f\n", pr, rc, f1))
if (f1 < 1.0) {
  cat("  Nicht perfekt -- das ist das ehrliche Ergebnis. Die vorherige In-Sample-Zahl\n")
  cat("  (Precision=Recall=F1=1.00, auf denselben Daten berichtet, die die Schwelle\n")
  cat("  festgelegt haben) war zirkulaer und sagte nichts ueber neue Fragen aus.\n")
} else {
  cat("  Auch out-of-sample perfekt auf diesem winzigen Set -- bei n=8 Fragen ist das\n")
  cat("  ein einzelner guenstiger Datenpunkt, kein Beleg fuer Generalisierung.\n")
}

# --- 3) Retriever + calibrated guard on example questions --------------------
cat("\n=== Retrieval + Guard auf Beispielfragen ===\n")
questions <- c(
  "Welcher Vasopressor wird bei Sepsis empfohlen?",
  "Was steht zur Nierenersatztherapie?",
  "Welche exakte Antibiotikadosis ist genannt?"  # not answered anywhere in the guideline
)
for (question in questions) {
  top <- retrieve(question, index, chunks)
  best_score <- top$score[1]
  cat(sprintf("\nFrage: %s\n", question))
  cat(sprintf("  Top-1-Aehnlichkeit (Cosinus): %.3f  (Schwelle: %.3f)\n", best_score, threshold))
  if (best_score < threshold) {
    cat("  -> Information in den bereitgestellten Leitlinien nicht enthalten (Score unter Schwelle).\n")
  } else {
    cat(sprintf("  -> Top-%d Chunks (zur Pruefung durch Menschen):\n", length(top$idx)))
    for (r in seq_along(top$idx)) {
      cat(sprintf("     %d. [Chunk %d, Score %.3f] %s...\n",
                  r, top$idx[r], top$score[r], substr(chunks[top$idx[r]], 1, 90)))
    }
  }
}
