"""
Gestion du Vector Store ChromaDB + Embeddings multilingues
"""
from pathlib import Path
from typing import List, Optional

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

from app.config import (
    CHROMA_DIR, CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K_RETRIEVAL
)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Charge le modèle d'embedding multilingue."""
    print(f"🔄 Chargement embedding : {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(documents: List[Document],
                       embedding_model: HuggingFaceEmbeddings) -> Chroma:
    """Construit et persiste le Vector Store ChromaDB."""
    print(f"🏗️  Construction ChromaDB ({len(documents)} chunks)...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_name=CHROMA_COLLECTION_NAME
    )
    print("✅ ChromaDB construit et persisté")
    return vectorstore


def load_vector_store(embedding_model: HuggingFaceEmbeddings) -> Optional[Chroma]:
    """Charge un Vector Store existant."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return None
    try:
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embedding_model,
            collection_name=CHROMA_COLLECTION_NAME
        )
        count = vectorstore._collection.count()
        if count == 0:
            return None
        print(f"✅ ChromaDB chargé ({count} vecteurs)")
        return vectorstore
    except Exception as e:
        print(f"⚠️ Erreur chargement ChromaDB: {e}")
        return None


#def get_retriever(vectorstore: Chroma):
def get_retriever(vectorstore, prefix: str = "query: "):
    """Crée un retriever LangChain depuis le vectorstore."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )

###  l'ajout du Cross-Encoder Reranker

def get_reranker_retriever(vectorstore, top_k_initial: int = 15, top_n_final: int = 8):
    """Retriever avec Cross-Encoder Reranker multilingue + MMR."""
    
    cross_encoder = CrossEncoder(
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    )
    
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k_initial, "fetch_k": 25, "lambda_mult": 0.5}
    )
    
    def rerank(query: str):
        docs = base_retriever.invoke(query)
        
        if not docs:
            return docs
        
        pairs = [[query, doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)
        
        scored_docs = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )
        
        return [doc for _, doc in scored_docs[:top_n_final]]
    
    return rerank

def build_hybrid_retriever(vectorstore, all_docs):
    """Crée un retriever hybride BM25 + vectoriel."""
    try:
        from rag.hybrid_retriever import HybridRetriever
        return HybridRetriever(vectorstore, all_docs)
    except Exception as e:
        print(f"⚠️ Hybrid retriever non disponible: {e}")
        return None
    
