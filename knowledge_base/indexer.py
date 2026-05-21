"""
Script d'indexation de la base de connaissances dans ChromaDB
Lancer une seule fois (ou après mise à jour des documents)

    python -m knowledge_base.indexer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_base.loader import load_all_documents, split_documents
from rag.vector_store import get_embedding_model, build_vector_store, load_vector_store
from app.config import CHROMA_DIR
import shutil


def index_knowledge_base(force_rebuild: bool = False):
    print("=" * 55)
    print("📚 INDEXATION DE LA BASE DE CONNAISSANCES")
    print("=" * 55)

    # Vérifie si déjà indexé
    embedding_model = get_embedding_model()
    if not force_rebuild:
        vs = load_vector_store(embedding_model)
        if vs is not None:
            print("✅ Base déjà indexée. Utilisez --rebuild pour forcer.")
            return vs

    # Rebuild
    if force_rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir()
        print("🗑️  Ancienne base supprimée")

    # Chargement des documents
    documents = load_all_documents()
    if not documents:
        print("❌ Aucun document trouvé dans knowledge_base/documents/")
        print("   Ajoutez vos fichiers CSV et PDF dans ce dossier")
        return None

    # Découpage en chunks
    chunks = split_documents(documents)

    # Construction du vectorstore
    vectorstore = build_vector_store(chunks, embedding_model)

    print("\n" + "=" * 55)
    print("✅ INDEXATION TERMINÉE")
    print("=" * 55)
    return vectorstore, chunks


if __name__ == "__main__":
    force = "--rebuild" in sys.argv
    index_knowledge_base(force_rebuild=force)
