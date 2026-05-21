"""
=============================================================
TEST 04 — Reranker (Cross-Encoder)
=============================================================
Objectif : Montrer comment le Cross-Encoder réordonne les
documents pour une question d'élève — les docs les plus
pertinents remontent en tête.
=============================================================
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from rag.vector_store import get_embedding_model, load_vector_store
from rag.hybrid_retriever import HybridRetriever


RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def test_reranker():
    print("=" * 65)
    print("  TEST 04 — Reranker Cross-Encoder")
    print(f"  Modèle : {RERANKER_MODEL}")
    print("=" * 65)

    print("\n  ⏳ Chargement du Cross-Encoder...")
    start = time.time()
    reranker = CrossEncoder(RERANKER_MODEL)
    print(f"  ✅ Chargé en {time.time() - start:.2f}s")

    embedding_model = get_embedding_model()
    vector_store = load_vector_store(embedding_model)

    if vector_store is None:
        raise ValueError("❌ ChromaDB introuvable ou vide. Construis d'abord le vector store.")

    data = vector_store.get(include=["documents", "metadatas"])

    all_docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]

    hybrid_retriever = HybridRetriever(
        vectorstore=vector_store,
        all_docs=all_docs
    )

    profil = "Bac SMA  — passionné d'informatique et d'IA"
    question = "J'ai un Bac Sciences Maths , je suis passionné d'informatique et d'intelligence artificielle, quelles filières sont disponibles pour moi dans les écoles publiques ?"

    print(f"\n  👤 Profil : {profil}")
    print(f"  📝 Question : \"{question}\"")

    documents = hybrid_retriever.retrieve(question)
    top_docs = documents[:8]

    if not top_docs:
        raise ValueError("❌ Aucun document retourné par le retriever hybride.")

    print("\n   AVANT Reranking (ordre du retriever) :")
    print(f"  {'-' * 55}")

    for i, doc in enumerate(top_docs):
        content = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i+1}] {content}...")

    start = time.time()
    pairs = [(question, doc.page_content) for doc in top_docs]
    scores = reranker.predict(pairs)
    rerank_time = time.time() - start

    scored_docs = sorted(
        zip(top_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n   APRÈS Reranking (réordonné par Cross-Encoder) :")
    print(f"  {'-' * 55}")

    for i, (doc, score) in enumerate(scored_docs):
        content = doc.page_content[:100].replace("\n", " ")
        source = os.path.basename(doc.metadata.get("source", "N/A"))
        print(f"    [{i+1}] Score: {score:.4f} | {content}...")
        print(f"         {source}")

    print("\n   Analyse du reranking :")
    print(f"  {'-' * 55}")
    print(f"    Temps de reranking : {rerank_time:.3f}s")
    print(f"    Meilleur score       : {max(scores):.4f}")
    print(f"    Pire score           : {min(scores):.4f}")
    print(f"    Score moyen          : {sum(scores)/len(scores):.4f}")

    original_ids = [id(d) for d in top_docs]
    reranked_ids = [id(d) for d, s in scored_docs]
    changed = original_ids != reranked_ids

    print(f"     Ordre modifié      : {'Oui ' if changed else 'Non'}")

    if changed:
        print("\n    Mouvements :")
        for new_pos, (doc, score) in enumerate(scored_docs):
            old_pos = original_ids.index(id(doc))
            if new_pos != old_pos:
                arrow = "⬆️" if new_pos < old_pos else "⬇️"
                print(f"      {arrow} Position {old_pos+1} → {new_pos+1} (score: {score:.4f})")

    print(f"\n{'=' * 65}")
    print("  ✅ TEST 04 TERMINÉ — Reranking validé")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    test_reranker()