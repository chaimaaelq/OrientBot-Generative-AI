import sys
import re
sys.path.insert(0, '.')
from rag.vector_store import get_embedding_model, load_vector_store

emb = get_embedding_model()
vs  = load_vector_store(emb)

question = "quels établissements se trouvent à Taroudant ?"
question_lower = question.lower()

mots_vides = {"quels", "quelles", "quel", "quelle", "sont", "dans",
              "pour", "avec", "les", "des", "une", "qui", "comment",
              "existe", "existent", "trouver", "trouve",
              "après", "apres", "faire", "peut", "avoir", "être"}

# Nouveau strip correct
mots = [re.sub(r'^[^a-zA-ZÀ-ÿ]+|[^a-zA-ZÀ-ÿ]+$', '', m)
        for m in question_lower.split()
        if len(m) > 4 and m not in mots_vides]
mots = [m for m in mots if m]

print("Mots extraits:", mots)

all_chromadb = vs.get()
print(f"Total docs ChromaDB: {len(all_chromadb['documents'])}")

for mot in mots:
    matching = [
        doc for doc in all_chromadb['documents']
        if mot.capitalize() in doc or mot in doc.lower()
    ]
    print(f"Mot '{mot}' → {len(matching)} docs trouvés")
    for d in matching[:2]:
        print(f"  → {d[:150]}")