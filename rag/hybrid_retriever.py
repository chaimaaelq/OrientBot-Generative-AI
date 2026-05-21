"""
Hybrid Search : BM25 (mots-clés) + Vectoriel (sémantique)
"""
import re
from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


def tokenize(text: str) -> List[str]:
    """Tokenisation simple pour le français."""
    text = text.lower()
    tokens = re.findall(r'[a-zA-ZÀ-ÿ0-9]+', text)
    return [t for t in tokens if len(t) > 2]


class HybridRetriever:
    def __init__(self, vectorstore, all_docs: List[Document],
                 k_vector: int = 10, k_bm25: int = 10, k_final: int = 6):
        self.vectorstore = vectorstore
        self.all_docs    = all_docs
        self.k_vector    = k_vector
        self.k_bm25      = k_bm25
        self.k_final     = k_final

        # Construire l'index BM25
        print("🔄 Construction index BM25...")
        tokenized = [tokenize(doc.page_content) for doc in all_docs]
        self.bm25 = BM25Okapi(tokenized)
        print(f"✅ BM25 indexé ({len(all_docs)} documents)")

    def retrieve(self, query: str) -> List[Document]:
        # ── Recherche vectorielle ─────────────────────────────────────────
        vector_docs = self.vectorstore.similarity_search(
            query, k=self.k_vector
        )

        # ── Recherche BM25 ────────────────────────────────────────────────
        query_tokens = tokenize(query)
        bm25_scores  = self.bm25.get_scores(query_tokens)

        # Top-K BM25
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:self.k_bm25]
        bm25_docs = [self.all_docs[i] for i in top_indices
                     if bm25_scores[i] > 0]

        # ── Fusion avec scores RRF ────────────────────────────────────────
        # RRF = Reciprocal Rank Fusion (standard en IR)
        scores = {}
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content[:80]
            scores[key] = scores.get(key, 0) + 1 / (rank + 60)

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:80]
            scores[key] = scores.get(key, 0) + 1 / (rank + 60)

        # Récupérer les docs uniques triés par score RRF
        seen  = set()
        fused = []
        all_candidates = vector_docs + bm25_docs
        for doc in sorted(
            all_candidates,
            key=lambda d: scores.get(d.page_content[:80], 0),
            reverse=True
        ):
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                fused.append(doc)

        return fused[:self.k_final]