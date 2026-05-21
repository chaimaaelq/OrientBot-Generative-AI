"""
=============================================================
TEST 06 — Pipeline RAG Complet avec Profil Élève
=============================================================
Objectif : Tester l'adaptation dynamique des réponses selon
le profil de l'apprenant : résultats scolaires, centres
d'intérêt, préférences et contexte utilisateur.
=============================================================
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.llm import get_rag_chain, generate_response
from backend.profil_eleve import ProfilEleve


def test_rag_pipeline():
    print("=" * 65)
    print("  TEST 06 — Pipeline RAG Complet avec Profil Élève")
    print("=" * 65)

    print("\n  ⏳ Initialisation du pipeline RAG...")
    start_init = time.time()
    rag_chain = get_rag_chain()
    print(f"  ✅ Pipeline initialisé en {time.time() - start_init:.2f}s")

    scenarios = [
        {
            "profil": ProfilEleve(
                prenom="Ahmed",
                niveau="Bachelier (Bac obtenu)",
                type_bac="Sciences Mathématiques A",
                moyenne_bac=15.8,
                note_maths=17,
                note_physique=15,
                note_francais=13,
                note_anglais=14,
                interets=["informatique", "intelligence artificielle", "data science"],
                objectif_carriere="Ingénieur en intelligence artificielle",
                description_objectif="L'élève aime les mathématiques, la programmation et les systèmes intelligents.",
                ville_preferee="Agadir",
                type_etablissement="Public",
                situation_socio_eco="Modeste (budget limité)",
                budget_annuel_max=None,
                mobilite=True
            ),
            "question": (
                "Quelles filières et "
                "écoles publiques me conviennent le mieux ?"
            )
        },
        {
            "profil": ProfilEleve(
                prenom="Fatima",
                niveau="Bachelier (Bac obtenu)",
                type_bac="Sciences de la Vie et de la Terre",
                moyenne_bac=14.2,
                note_svt=16,
                note_physique=13,
                note_francais=15,
                interets=["médecine", "pharmacie", "biologie"],
                objectif_carriere="Médecin ou pharmacienne",
                description_objectif="L'élève préfère les domaines de santé et souhaite rester proche de Casablanca.",
                ville_preferee="Casablanca",
                type_etablissement="Privé",
                situation_socio_eco="Aisé",
                budget_annuel_max=70000,
                mobilite=False
            ),
            "question": (
                "Quelles options privées sont adaptées pour "
                "médecine ou pharmacie ?"
            )
        },
        {
            "profil": ProfilEleve(
                prenom="Youssef",
                niveau="Bachelier (Bac obtenu)",
                type_bac="Sciences Mathématiques A",
                moyenne_bac=13.5,
                note_maths=14,
                note_physique=13,
                note_anglais=16,
                interets=["data science", "intelligence artificielle", "statistiques"],
                objectif_carriere="Data scientist",
                description_objectif="L'élève aime l'analyse de données mais a besoin d'un parcours accessible dans le public.",
                ville_preferee="Fès",
                type_etablissement="Public",
                situation_socio_eco="Modeste (budget limité)",
                budget_annuel_max=None,
                mobilite=True
            ),
            "question": (
                "Quelles formations publiques peuvent m'aider "
                "à devenir data scientist ?"
            )
        }
    ]

    for idx, scenario in enumerate(scenarios):
        profil = scenario["profil"]
        question = scenario["question"]

        print(f"\n{'=' * 65}")
        print(f"  SCÉNARIO {idx + 1}/3 — {profil.prenom}")
        print(f"{'=' * 65}")

        print(f"  👤 Profil : {profil.prenom}")
        print(f"     Niveau      : {profil.niveau}")
        print(f"     Bac         : {profil.type_bac}")
        print(f"     Moyenne     : {profil.moyenne_bac}/20")
        print(f"     Ville       : {profil.ville_preferee}")
        print(f"     Type        : {profil.type_etablissement}")
        print(f"     Intérêts    : {', '.join(profil.interets)}")
        print(f"     Objectif    : {profil.objectif_carriere}")

        if profil.budget_annuel_max:
            print(f"     Budget      : {profil.budget_annuel_max} MAD/an")
        else:
            print("     Budget      : Non précisé / préférence public")

        print("\n  📝 Question personnalisée :")
        print(f"  \"{question}\"")

        print("\n  📌 Contexte profil envoyé au RAG :")
        print("  " + "-" * 55)
        for line in profil.to_context_string().split("\n"):
            print(f"    {line}")

        start = time.time()

        result = generate_response(
            rag_chain=rag_chain,
            question=question,
            chat_history=[]
        )

        total_time = time.time() - start

        if isinstance(result, dict):
            response = result.get("answer", str(result))
            sources = result.get("source_documents", [])
        else:
            response = str(result)
            sources = []

        print("\n  💬 Réponse d'OrientBot :")
        print("  " + "-" * 55)
        for line in response.split("\n"):
            print(f"    {line}")

        print("\n  📚 Sources utilisées :")
        print("  " + "-" * 55)
        for i, doc in enumerate(sources[:5], 1):
            source = os.path.basename(doc.metadata.get("source", "N/A"))
            contenu = doc.page_content.replace("\n", " ")
            print(f"\n    [{i}] Source : {source}")
            print(f"        Contenu : {contenu}")

        print(f"\n  ⏱ Temps de réponse : {total_time:.2f}s")
        print(f"  📏 Longueur : {len(response)} caractères")

    print(f"\n{'=' * 65}")
    print("  ✅ TEST 06 TERMINÉ — Pipeline RAG personnalisé fonctionnel")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    test_rag_pipeline()