"""Debug complet — vérifie ce qui arrive au LLM"""
import sys
sys.path.insert(0, '.')

from rag.vector_store import get_embedding_model, load_vector_store
from rag.llm import get_llm, build_rag_chain, generate_response

emb = get_embedding_model()
vs = load_vector_store(emb)

if not vs:
    print("❌ ChromaDB vide")
    sys.exit(1)

llm = get_llm()

# Simuler le profil "Les deux" + Agadir + Commerce
profil_context = """PROFIL DE L'ÉLÈVE : Chaimaa
- Niveau : Bachelier (Bac obtenu)
- Type de Bac : Sciences Économiques et Gestion
- Moyenne Bac : 16.0/20
- Centres d'intérêt : Commerce & Gestion
- Ville souhaitée : Agadir
- Type d'établissement : Les deux (pas de préférence)"""

from rag.vector_store import get_retriever
retr = get_retriever(vs)
chain = build_rag_chain(retr, llm, profil_context, vs)

question = "Quelles filières correspondent à mon profil ?"

print("\n" + "="*60)
print(f"Question: {question}")
print(f"Profil: Agadir, SEG, Les deux")
print("="*60)

result = generate_response(chain, question, [])
print("\n🧭 RÉPONSE:")
print(result["answer"])
print(f"\n📚 Nombre de docs sources: {len(result['source_documents'])}")
print("\n=== SOURCES UTILISÉES ===")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"[{i}] {doc.metadata.get('source','?')}")
    print(f"    {doc.page_content[:150]}\n")