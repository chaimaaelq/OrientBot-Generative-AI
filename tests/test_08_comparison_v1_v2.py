"""
=============================================================
TEST 08 — Comparaison RAG Basique vs RAG Avancé
=============================================================
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.llm import get_rag_chain, generate_response
from rag.vector_store import get_embedding_model, load_vector_store
from rag.hybrid_retriever import HybridRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def get_hybrid_retriever():
    embedding_model = get_embedding_model()
    vector_store = load_vector_store(embedding_model)

    if vector_store is None:
        raise ValueError("❌ ChromaDB introuvable")

    data = vector_store.get(include=["documents", "metadatas"])

    all_docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]

    hybrid = HybridRetriever(vectorstore=vector_store, all_docs=all_docs)
    return vector_store, hybrid


# =====================
# V1 : Basique
# =====================
def rag_v1(rag_chain, question):
    rag_chain["profil_context"] = "Profil non spécifié"

    result = generate_response(
        rag_chain=rag_chain,
        question=question,
        chat_history=[]
    )

    return result.get("answer", ""), result.get("source_documents", [])


# =====================
# V2 : Avancé
# =====================
def rag_v2(rag_chain, hybrid, reranker, question, profil_context):
    # 1. Hybrid retrieval
    docs = hybrid.retrieve(question)

    # 2. Reranking
    pairs = [(question, d.page_content) for d in docs[:10]]
    scores = reranker.predict(pairs)

    docs = [
        d for d, s in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    ][:6]

    # 3. Inject profil
    rag_chain["profil_context"] = profil_context

    result = generate_response(
        rag_chain=rag_chain,
        question=question,
        chat_history=[]
    )

    return result.get("answer", ""), docs


def test_comparison():
    print("=" * 70)
    print("TEST 08 — RAG Basique vs Avancé")
    print("=" * 70)

    rag_chain = get_rag_chain()
    vector_store, hybrid = get_hybrid_retriever()
    reranker = CrossEncoder(RERANKER_MODEL)

    profil = """
Ville : Agadir
Bac : Sciences Mathématiques A
Type : Public
Intérêts : informatique, IA
"""

    question = (
        "J'ai un Bac Sciences Maths A, je suis à Agadir et j'adore "
        "l'informatique et l'IA. Quelles écoles d'ingénieurs publiques "
        "je peux intégrer ?"
    )

    print("\n👤 Profil :", profil.strip())
    print("📝 Question :", question)

    # =====================
    # V1
    # =====================
    print("\n🔴 V1 — Basique")
    start = time.time()

    resp1, docs1 = rag_v1(rag_chain, question)

    t1 = time.time() - start

    print("\nRéponse V1 :")
    print(resp1)

    # =====================
    # V2
    # =====================
    print("\n🟢 V2 — Avancé")
    start = time.time()

    resp2, docs2 = rag_v2(rag_chain, hybrid, reranker, question, profil)

    t2 = time.time() - start

    print("\nRéponse V2 :")
    print(resp2)

    # =====================
    # COMPARAISON
    # =====================
    print("\n" + "=" * 70)
    print("TABLEAU COMPARATIF")
    print("=" * 70)

    print(f"{'Critère':<30} {'V1':<15} {'V2':<15}")
    print("-" * 60)
    print(f"{'Retrieval':<30} {'Vectoriel':<15} {'Hybride':<15}")
    print(f"{'Profil':<30} {'Non':<15} {'Oui':<15}")
    print(f"{'Reranker':<30} {'Non':<15} {'Oui':<15}")
    print(f"{'Temps':<30} {t1:.2f}s{'':<10} {t2:.2f}s")

    print("\n✅ TEST TERMINÉ")


if __name__ == "__main__":
    test_comparison()