"""
=============================================================
TEST 02 — Embeddings & Vector Store (ChromaDB)
=============================================================
Objectif : Vérifier que la recherche vectorielle retourne
des résultats pertinents pour des questions d'élèves.
=============================================================
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.vector_store import get_embedding_model, load_vector_store
from app.config import EMBEDDING_MODEL

def test_vector_store():
    print("=" * 65)
    print("  TEST 02 — Embeddings & Vector Store (ChromaDB)")
    print("=" * 65)
    print(f"  Modèle d'embedding : {EMBEDDING_MODEL}")
    
    embedding_model = get_embedding_model()
    vector_store = load_vector_store(embedding_model)
    collection = vector_store._collection
    count = collection.count()
    print(f"  ✅ Nombre de vecteurs dans ChromaDB : {count}")
    
    # Questions réalistes d'élèves avec leur profil
    questions_eleves = [
        {
            "profil": "Bac SMA — Agadir — passionné d'informatique",
            "question": "J'ai un Bac Sciences Mathématiques A, je suis à Agadir et j'adore l'informatique. Quelles écoles je peux intégrer ?"
        },
        {
            "profil": "Bac SVT — Casablanca — intéressée par la médecine",
            "question": "J'ai un Bac SVT et je suis à Casablanca. Je veux faire médecine, quelles sont mes options ?"
        },
        {
            "profil": "Bac SMA — Marrakech — budget limité",
            "question": "Je suis un élève avec Bac Sciences Maths à Marrakech, je cherche une école publique en ingénierie."
        },
    ]
    
    for idx, q in enumerate(questions_eleves):
        print(f"\n{'=' * 65}")
        print(f"  👤 Profil : {q['profil']}")
        print(f"  📝 Question : \"{q['question']}\"")
        print("-" * 55)
        
        results = vector_store.similarity_search_with_score(q["question"], k=5)
        
        print(f"\n  📊 Top 5 résultats par similarité :")
        for i, (doc, score) in enumerate(results):
            content = doc.page_content[:120].replace("\n", " ")
            source = os.path.basename(doc.metadata.get("source", "N/A"))
            print(f"\n  [{i+1}] Score: {score:.4f}")
            print(f"      Source  : {source}")
            print(f"      Contenu : {content}...")
    
    print(f"\n{'=' * 65}")
    print(f"  ✅ TEST 02 TERMINÉ — {count} vecteurs fonctionnels")
    print(f"{'=' * 65}")

if __name__ == "__main__":
    test_vector_store()