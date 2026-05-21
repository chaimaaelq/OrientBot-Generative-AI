"""Test recherche établissements privés"""
import sys
sys.path.insert(0, '.')

from rag.vector_store import get_embedding_model, load_vector_store

emb = get_embedding_model()
vs = load_vector_store(emb)

if not vs:
    print("❌ ChromaDB vide")
    sys.exit(1)

# Test 1 — recherche directe privé Agadir
print("\n=== TEST 1 : privé Agadir commerce ===")
docs = vs.similarity_search("établissement privé Agadir commerce gestion frais scolarité", k=5)
for i, d in enumerate(docs, 1):
    print(f"[{i}] {d.metadata.get('source','?')}")
    print(f"    {d.page_content[:150]}\n")

# Test 2 — recherche par mot "privé" dans le contenu
print("\n=== TEST 2 : contenu contenant Privé + Agadir ===")
all_data = vs.get()
priv_agadir = [
    (doc, meta) for doc, meta in zip(all_data['documents'], all_data['metadatas'])
    if 'privé' in doc.lower() or 'prive' in doc.lower() or 'Privé' in doc
    if 'agadir' in doc.lower()
]
print(f"Trouvés : {len(priv_agadir)}")
for doc, meta in priv_agadir[:5]:
    print(f"  Source: {meta.get('source','?')}")
    print(f"  {doc[:150]}\n")

# Test 3 — tous les privés dans la base
print("\n=== TEST 3 : tous les privés dans ChromaDB ===")
tous_prives = [
    (doc, meta) for doc, meta in zip(all_data['documents'], all_data['metadatas'])
    if 'privé' in doc.lower() or 'Privé' in doc
]
print(f"Total privés : {len(tous_prives)}")

# Test 4 — chercher Universiapolis et EMAA spécifiquement
print("\n=== TEST 4 : Universiapolis + EMAA dans ChromaDB ===")
for mot in ["universiapolis", "emaa", "isiam", "polytechnique agadir"]:
    matching = [
        (doc, meta) for doc, meta in zip(all_data['documents'], all_data['metadatas'])
        if mot in doc.lower()
    ]
    print(f"'{mot}' → {len(matching)} docs")
    for doc, meta in matching[:2]:
        print(f"  Source: {meta.get('source','?')}")
        print(f"  {doc[:200]}\n")