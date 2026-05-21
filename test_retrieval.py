"""Test rapide du retrieval - diagnostic"""
import sys
sys.path.insert(0, '.')

from rag.vector_store import get_embedding_model, load_vector_store

emb = get_embedding_model()
vs = load_vector_store(emb)

if not vs:
    print("❌ ChromaDB vide ou introuvable")
    sys.exit(1)

query = "établissements Intelligence Artificielle Maroc"
docs = vs.similarity_search(query, k=12)

print(f"\n{'='*60}")
print(f"Docs trouvés: {len(docs)}")
print(f"{'='*60}\n")

for i, doc in enumerate(docs, 1):
    print(f"[{i}] {doc.metadata.get('source','?')}")
    print(f"    {doc.page_content[:150]}")
    print()