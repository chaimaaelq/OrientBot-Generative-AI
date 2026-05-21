"""
=============================================================
TEST 03 — Retriever Hybride (BM25 + Vectoriel)
=============================================================
Objectif : Comparer vectoriel seul vs hybride sur des
questions réelles d'élèves marocains.
=============================================================
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.vector_store import get_embedding_model, load_vector_store
from rag.hybrid_retriever import HybridRetriever
from langchain_core.documents import Document

def test_hybrid_retriever():
    print("=" * 65)
    print("  TEST 03 — Retriever Hybride (BM25 + Vectoriel)")
    print("=" * 65)
    
    embedding_model = get_embedding_model()
    vector_store = load_vector_store(embedding_model)

    data = vector_store.get(include=["documents", "metadatas"])

    all_docs = [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]

    hybrid_retriever = HybridRetriever(vectorstore=vector_store, all_docs=all_docs)
    
    # Questions d'élèves réalistes
    questions_eleves = [
        {
            "profil": "Bac SMA — Agadir — informatique",
            "question": "J'ai un Bac Sciences Maths A et je suis à Agadir. Je veux faire Génie Informatique. Quelles ecoles je peux viser ?"
        },
        {
            "profil": "Bac SVT — Tanger — biologie",
            "question": "Je suis bachelier SVT à Tanger, j'aime la biologie. Quelles filières sont disponibles pour moi dans les écoles publiques ?"
        },
        {
            "profil": "Bac SMA — IA et data science",
            "question": "J'ai eu mon Bac Sciences Maths et je m'intéresse à l'intelligence artificielle. Quels établissements au Maroc proposent l'IA ?"
        },
    ]
    
    for q_idx, q in enumerate(questions_eleves):
        print(f"\n{'=' * 65}")
        print(f"  👤 Profil : {q['profil']}")
        print(f"  📝 Question : \"{q['question']}\"")
        print(f"{'=' * 65}")
        
        # --- Recherche Vectorielle seule ---
        print(f"\n  🔵 Recherche VECTORIELLE seule :")
        print(f"  {'-' * 50}")
        start = time.time()
        vec_results = vector_store.similarity_search(q["question"], k=5)
        vec_time = time.time() - start
        
        for i, doc in enumerate(vec_results[:3]):
            content = doc.page_content[:100].replace("\n", " ")
            print(f"    [{i+1}] {content}...")
        print(f"    ⏱ Temps : {vec_time:.3f}s | {len(vec_results)} résultats")
        
        # --- Recherche Hybride ---
        print(f"\n  🟢 Recherche HYBRIDE (BM25 + Vectoriel) :")
        print(f"  {'-' * 50}")
        start = time.time()
        hybrid_results = hybrid_retriever.retrieve(q["question"])
        hybrid_time = time.time() - start
        
        for i, doc in enumerate(hybrid_results[:5]):
            content = doc.page_content[:100].replace("\n", " ")
            source = os.path.basename(doc.metadata.get("source", "N/A"))
            print(f"    [{i+1}] {content}...")
            print(f"        📄 Source: {source}")
        print(f"    ⏱ Temps : {hybrid_time:.3f}s | {len(hybrid_results)} résultats")
        
        # --- Comparaison ---
        print(f"\n  📊 Comparaison :")
        print(f"  {'-' * 50}")
        vec_set = set(d.page_content[:80] for d in vec_results[:5])
        hyb_set = set(d.page_content[:80] for d in hybrid_results[:5])
        overlap = vec_set & hyb_set
        unique_bm25 = hyb_set - vec_set
        
        print(f"    Documents communs (top 5)         : {len(overlap)}")
        print(f"    Nouveaux docs grâce à BM25         : {len(unique_bm25)}")
        print(f"    → BM25 apporte des résultats {'supplémentaires ✅' if unique_bm25 else 'identiques'}")
    
    print(f"\n{'=' * 65}")
    print(f"  ✅ TEST 03 TERMINÉ — Retriever hybride validé")
    print(f"{'=' * 65}")

if __name__ == "__main__":
    test_hybrid_retriever()