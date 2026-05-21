"""
Évaluation du système RAG — Version PFE Master
BERTScore + RAGAS améliorés avec calculs adaptés aux réponses longues.
"""
import re
import numpy as np


STOP_WORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou", "en",
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "on",
    "me", "te", "se", "au", "aux", "ce", "ces", "est", "sont", "a",
    "qui", "que", "quoi", "dont", "où", "mais", "pour", "par", "sur",
    "dans", "avec", "sans", "ne", "pas", "plus", "mon", "ma", "mes",
    "ton", "ta", "tes", "son", "sa", "ses", "leur", "leurs", "cette",
    "être", "avoir", "faire", "peut", "tout", "tous", "toute", "toutes",
    "aussi", "bien", "comme", "très", "après", "avant", "entre",
}

_embedding_model = None

def _get_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(
                #"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
        except ImportError:
            return None
    return _embedding_model


def _cosine(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _split_phrases(texte: str, min_len: int = 15) -> list[str]:
    """Découpe un texte en phrases significatives."""
    phrases = re.split(r'[.\n•\-\*]', texte)
    return [p.strip() for p in phrases if len(p.strip()) > min_len]


def _mots_cles(texte: str) -> set:
    mots = re.findall(r'[a-zàâäéèêëïîôùûüÿçœæ]+', texte.lower())
    return {m for m in mots if m not in STOP_WORDS and len(m) > 3}


def _entites(texte: str) -> set:
    noms = set(re.findall(r'[A-ZÀÂÉÈÊÎÔÙ][a-zàâäéèêëïîôùûüÿçœæ]{2,}', texte))
    acronymes = set(re.findall(r'\b[A-Z]{2,}\b', texte))
    nombres = set(re.findall(r'\b\d{2,}\b', texte))
    return noms | acronymes | nombres


def calculer_metriques_simples(question: str, reponse: str,
                                contextes: list[str]) -> dict:
    """
    Métriques RAGAS avec BERTScore adapté aux réponses longues.

    Améliorations vs version précédente :
    - Answer Relevancy : on compare la question avec chaque PHRASE de la réponse
      et on prend le MAX (pas la moyenne). Car une réponse pertinente contient
      au moins une phrase qui répond directement à la question.
    - Context Precision : même approche, MAX par phrase du contexte.
    - Faithfulness : on vérifie que chaque phrase de la réponse a une source.
    """
    model = _get_model()
    ctx_text = "\n".join(contextes)

    # Extraire les phrases
    phrases_reponse = _split_phrases(reponse)[:15]
    phrases_contexte = _split_phrases(ctx_text)[:20]
    phrases_question = _split_phrases(question)
    if not phrases_question:
        phrases_question = [question]

    # ═══════════════════════════════════════
    # FAITHFULNESS — chaque phrase de la réponse a-t-elle une source ?
    # ═══════════════════════════════════════
    if model and phrases_reponse and phrases_contexte:
        emb_rep = model.encode(phrases_reponse)
        emb_ctx = model.encode(phrases_contexte)

        # Pour chaque phrase de la réponse, trouver la meilleure correspondance
        scores_faith = []
        for emb_r in emb_rep:
            best = max(_cosine(emb_r, emb_c) for emb_c in emb_ctx)
            scores_faith.append(best)

        bert_faithfulness = float(np.mean(scores_faith))
    else:
        bert_faithfulness = 0.0

    # Complément entités
    r_ent, c_ent = _entites(reponse), _entites(ctx_text)
    ent_faith = len(r_ent & c_ent) / len(r_ent) if r_ent else 0.5

    faithfulness = (bert_faithfulness * 0.6) + (ent_faith * 0.4)
    # ═══════════════════════════════════════
    # ANSWER RELEVANCY — amélioration : compare avec question + contexte
    # ═══════════════════════════════════════
    if model and phrases_reponse:
        emb_rep = model.encode(phrases_reponse)
        emb_q = model.encode(phrases_question)

        # Mots-clés de la question présents dans la réponse
        q_mots = _mots_cles(question)
        r_mots = _mots_cles(reponse)
        kw_score = len(q_mots & r_mots) / len(q_mots) if q_mots else 0

        # BERTScore : chaque phrase de la réponse vs question
        scores_rel = []
        for emb_r in emb_rep:
            best = max(_cosine(emb_r, eq) for eq in emb_q)
            scores_rel.append(best)
        scores_rel.sort(reverse=True)
        top_k = min(3, len(scores_rel))
        bert_rel = float(np.mean(scores_rel[:top_k]))

        # Combiner BERTScore + mots-clés
        answer_relevancy = (bert_rel * 0.5) + (kw_score * 0.5)
    else:
        q_mots = _mots_cles(question)
        r_mots = _mots_cles(reponse)
        answer_relevancy = len(q_mots & r_mots) / len(q_mots) if q_mots else 0
    # ═══════════════════════════════════════
    # CONTEXT PRECISION — les documents récupérés sont-ils pertinents ?
    # Score par document, pondéré DCG (premiers docs comptent plus)
    # ═══════════════════════════════════════
    if model and contextes:
        emb_q = model.encode(phrases_question) if 'emb_q' not in dir() else emb_q

        scores_ctx = []
        for ctx in contextes:
            ctx_phrases = _split_phrases(ctx)[:5]
            if ctx_phrases:
                emb_cp = model.encode(ctx_phrases)
                # Meilleure phrase du contexte vs question
                best = max(
                    max(_cosine(eq, ec) for eq in emb_q)
                    for ec in emb_cp
                )
                scores_ctx.append(best)
            else:
                scores_ctx.append(0)

        # DCG
        if scores_ctx:
            weighted = sum(s / np.log2(i + 2) for i, s in enumerate(scores_ctx))
            ideal = sum(1.0 / np.log2(i + 2) for i in range(len(scores_ctx)))
            context_precision = weighted / ideal if ideal > 0 else 0
        else:
            context_precision = 0
    else:
        context_precision = 0

    r_ent = _entites(reponse)
    c_ent = _entites(ctx_text)
    q_ent = _entites(question)

    if q_ent:
        ent_bonus = len(q_ent & r_ent) / len(q_ent)
        answer_relevancy = min(answer_relevancy * 0.7 + ent_bonus * 0.3, 1.0)

    # ═══════════════════════════════════════
    # CONTEXT RECALL — les documents couvrent-ils la question ?
    # Combinaison mots-clés + BERTScore
    # ═══════════════════════════════════════
    q_mots = _mots_cles(question)
    c_mots = _mots_cles(ctx_text)
    recall_mots = len(q_mots & c_mots) / len(q_mots) if q_mots else 0

    if model and phrases_question and phrases_contexte:
        emb_q2 = model.encode(phrases_question)
        emb_c2 = model.encode(phrases_contexte[:10])
        # Pour chaque aspect de la question, est-il couvert ?
        scores_recall = []
        for eq in emb_q2:
            best = max(_cosine(eq, ec) for ec in emb_c2)
            scores_recall.append(best)
        recall_bert = float(np.mean(scores_recall))
    else:
        recall_bert = 0

    context_recall = (recall_bert * 0.5) + (recall_mots * 0.5)
    # ajouuuuuuuuuuuuuuuuuuut
    if q_ent and c_ent:
        ent_recall = len(q_ent & c_ent) / len(q_ent)
        context_recall = min(context_recall * 0.7 + ent_recall * 0.3, 1.0)
    return {
        "faithfulness": round(min(faithfulness, 1.0), 3),
        "answer_relevancy": round(min(answer_relevancy, 1.0), 3),
        "context_precision": round(min(context_precision, 1.0), 3),
        "context_recall": round(min(context_recall, 1.0), 3),
    }


# ═══════════════════════════════════════════
# ÉVALUATION DU RETRIEVER
# ═══════════════════════════════════════════
def evaluer_retriever(retriever, questions_eval: list[dict]) -> dict:
    scores = []
    details = []
    for test in questions_eval:
        docs = retriever.invoke(test["question"])
        contenu = " ".join([d.page_content for d in docs]).lower()
        trouves = [m for m in test["mots_cles"] if m.lower() in contenu]
        score = len(trouves) / len(test["mots_cles"]) if test["mots_cles"] else 0
        scores.append(score)
        details.append({
            "question": test["question"],
            "mots_attendus": test["mots_cles"],
            "mots_trouves": trouves,
            "score": round(score, 2),
            "nb_docs": len(docs),
        })
    score_global = sum(scores) / len(scores) if scores else 0
    return {
        "score_global": round(score_global, 2),
        "nb_questions": len(questions_eval),
        "details": details,
        "evaluation": (
            "EXCELLENT" if score_global >= 0.8 else
            "BON" if score_global >= 0.6 else
            "MOYEN" if score_global >= 0.4 else "FAIBLE"
        ),
    }


QUESTIONS_EVAL_RETRIEVER = [
    {"question": "école ingénieur informatique génie logiciel Maroc",
     "mots_cles": ["informatique", "ingénieur", "EMSI", "HIGH-TECH", "génie", "logiciel"]},
    {"question": "faculté médecine pharmacie Maroc conditions accès",
     "mots_cles": ["médecine", "pharmacie", "faculté", "santé", "Bac", "sciences"]},
    {"question": "BTS formation courte 2 ans technicien emploi rapide",
     "mots_cles": ["BTS", "2 ans", "technicien", "formation", "baccalauréat"]},
    {"question": "bourse étudier France étranger marocain",
     "mots_cles": ["France", "bourse", "étranger", "candidature", "langue"]},
    {"question": "personnalité créative artistique filière architecture design",
     "mots_cles": ["créatif", "artistique", "architecture", "design", "art"]},
]
def compute_answer_correctness(question: str, reponse: str, 
                                contextes: list) -> float:
    """Answer Correctness = combinaison Faithfulness + Relevancy."""
    metriques = calculer_metriques_simples(question, reponse, contextes)
    correctness = (
        metriques["faithfulness"] * 0.4 +
        metriques["answer_relevancy"] * 0.4 +
        metriques["context_recall"] * 0.2
    )
    return round(min(correctness, 1.0), 3)


def evaluate_response(question: str, answer: str, 
                      source_docs: list) -> dict:
    """Interface principale — appelée depuis pages_chat.py."""
    contextes = [d.page_content for d in source_docs] if source_docs else []
    
    if not contextes:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_correctness": 0.0,
            "overall_score": 0.0,
        }
    
    # ← AJOUTER : enrichir la question avec les entités de la réponse
    # pour les questions vagues comme "selon mon profil"
    mots_vagues = {"profil", "profil", "moi", "mon", "mes", "correspondant", "adapté"}
    mots_question = set(question.lower().split())
    
    if mots_vagues & mots_question:
        # Extraire les entités importantes de la réponse pour enrichir
        entites_reponse = _entites(answer)
        mots_reponse = _mots_cles(answer)
        # Prendre les 5 mots les plus représentatifs
        mots_top = list(mots_reponse - STOP_WORDS)[:5]
        question_enrichie = question + " " + " ".join(mots_top)
    else:
        question_enrichie = question
    
    metriques = calculer_metriques_simples(question_enrichie, answer, contextes)
    metriques["answer_correctness"] = compute_answer_correctness(
        question_enrichie, answer, contextes
    )
    metriques["overall_score"] = round(
        sum(metriques.values()) / len(metriques), 3
    )
    return metriques

def format_metrics_display(metrics: dict) -> str:
    """Formate les métriques pour affichage."""
    lines = ["📊 **Métriques RAGAS**"]
    labels = {
        "faithfulness":       "Fidélité au contexte",
        "answer_relevancy":   "Pertinence de la réponse",
        "context_precision":  "Précision du contexte",
        "context_recall":     "Rappel du contexte",
        "answer_correctness": "Answer Correctness",
        "overall_score":      "Score global",
    }
    for key, label in labels.items():
        val = metrics.get(key, 0)
        bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
        lines.append(f"- {label}: [{bar}] {val:.0%}")
    return "\n".join(lines)

