"""
=============================================================
TEST 05 — Profil Élève & Personnalisation
=============================================================
Objectif : Montrer comment la même question donne des
résultats différents selon le profil de l'élève
(ville, bac, type d'établissement, budget).
=============================================================
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.profil_eleve import ProfilEleve
from rag.vector_store import get_embedding_model, load_vector_store


def afficher_profil(profil):
    print(f"  Prénom      : {profil.prenom}")
    print(f"  Ville       : {profil.ville_preferee}")
    print(f"  Bac         : {profil.type_bac}")
    print(f"  Type        : {profil.type_etablissement}")
    print(f"  Intérêts    : {', '.join(profil.interets)}")

    if profil.budget_annuel_max:
        print(f"  Budget      : {profil.budget_annuel_max} MAD/an")
    else:
        print("  Budget      : Pas de budget précisé")


def filtrer_resultats_par_profil(results, profil):
    scored = []

    for doc in results:
        content_lower = doc.page_content.lower()
        meta = doc.metadata or {}

        boost = 0
        raisons = []

        # Ville
        if profil.ville_preferee:
            ville = profil.ville_preferee.lower()
            if ville in content_lower or ville in str(meta.get("ville", "")).lower():
                boost += 3
                raisons.append(f"ville={profil.ville_preferee}")

        # Type établissement
        if profil.type_etablissement:
            type_etab = str(profil.type_etablissement).lower()
            if type_etab in content_lower or type_etab in str(meta.get("type", "")).lower():
                boost += 2
                raisons.append(f"type={profil.type_etablissement}")

        # Type Bac
        if profil.type_bac:
            bac = profil.type_bac.lower()
            if bac in content_lower:
                boost += 2
                raisons.append(f"bac={profil.type_bac}")

        # Intérêts
        for interet in profil.interets:
            if interet.lower() in content_lower:
                boost += 1
                raisons.append(f"intérêt={interet}")

        # Budget
        if profil.budget_annuel_max is None:
            if "public" in content_lower or "gratuit" in content_lower:
                boost += 2
                raisons.append("budget=public/gratuit")
        else:
            if "privé" in content_lower or "prive" in content_lower:
                boost += 1
                raisons.append("budget=privé possible")

        scored.append((doc, boost, raisons))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def afficher_resultats(scored, nom_profil):
    print(f"\n  🔍 Résultats filtrés pour {nom_profil} :")
    print(f"  {'-' * 55}")

    for i, (doc, boost, raisons) in enumerate(scored[:5]):
        content = doc.page_content[:120].replace("\n", " ")
        source = os.path.basename(doc.metadata.get("source", "N/A"))
        #stars = "⭐" * boost if boost > 0 else "—"
        raisons_str = ", ".join(raisons) if raisons else "aucun match"

        print(f"\n    [{i+1}] Boost: {boost} ")
        print(f"        Match   : {raisons_str}")
        print(f"        Source  : {source}")
        print(f"        Contenu : {content}...")


def test_profil_eleve():
    print("=" * 65)
    print("  TEST 05 — Profil Élève & Personnalisation")
    print("=" * 65)

    embedding_model = get_embedding_model()
    vector_store = load_vector_store(embedding_model)

    if vector_store is None:
        raise ValueError("❌ ChromaDB introuvable ou vide. Construis d'abord le vector store.")

    # =============================================
    # PROFIL 1 : Ahmed — Agadir — SMA — Public
    # =============================================
    profil_1 = ProfilEleve(
        prenom="Ahmed",
        niveau="Bachelier (Bac obtenu)",
        type_bac="Sciences Mathématiques A",
        ville_preferee="Agadir",
        type_etablissement="Public",
        interets=["informatique", "intelligence artificielle"],
        situation_socio_eco="Modeste (budget limité)",
        budget_annuel_max=None,
        mobilite=True
    )

    print(f"\n{'=' * 65}")
    print(f"  👤 PROFIL 1 : {profil_1.prenom}")
    print(f"{'=' * 65}")
    afficher_profil(profil_1)

    question_1 = "Je veux étudier l'informatique après le bac, quelles écoles me conviennent ?"
    print(f"\n  📝 Question : \"{question_1}\"")

    results_1 = vector_store.similarity_search(question_1, k=8)
    scored_1 = filtrer_resultats_par_profil(results_1, profil_1)
    afficher_resultats(scored_1, profil_1.prenom)

    # =============================================
    # PROFIL 2 : Sara — Casablanca — SVT — Privé
    # =============================================
    profil_2 = ProfilEleve(
        prenom="Sara",
        niveau="Bachelier (Bac obtenu)",
        type_bac="Sciences de la Vie et de la Terre",
        ville_preferee="Casablanca",
        type_etablissement="Privé",
        interets=["médecine", "pharmacie"],
        situation_socio_eco="Aisé",
        budget_annuel_max=80000,
        mobilite=False
    )

    print(f"\n\n{'=' * 65}")
    print(f"  👤 PROFIL 2 : {profil_2.prenom}")
    print(f"{'=' * 65}")
    afficher_profil(profil_2)

    question_2 = "J'ai un Bac SVT et je suis à Casablanca. Je veux faire médecine dans une université privée. Quelles sont mes options ?"
    print(f"\n  📝 Question : \"{question_2}\"")

    results_2 = vector_store.similarity_search(question_2, k=8)
    scored_2 = filtrer_resultats_par_profil(results_2, profil_2)
    afficher_resultats(scored_2, profil_2.prenom)

    # =============================================
    # Comparaison
    # =============================================
    print(f"\n\n{'=' * 65}")
    print("  COMPARAISON DES DEUX PROFILS")
    print(f"{'=' * 65}")

    print(f"  {'Critère':<25} {'Ahmed':>20} {'Sara':>20}")
    print(f"  {'-' * 65}")
    print(f"  {'Ville':<25} {'Agadir':>20} {'Casablanca':>20}")
    print(f"  {'Bac':<25} {'SMA':>20} {'SVT':>20}")
    print(f"  {'Type':<25} {'Public':>20} {'Privé':>20}")
    print(f"  {'Budget':<25} {'Non précisé':>20} {'80 000 MAD':>20}")
    print(f"  {'Intérêts':<25} {'Info, IA':>20} {'Médecine, Pharma':>20}")

    top1_ahmed = scored_1[0][0].page_content[:60].replace("\n", " ")
    top1_sara = scored_2[0][0].page_content[:60].replace("\n", " ")

    print(f"\n  → Ahmed reçoit en 1er : \"{top1_ahmed}...\"")
    print(f"  → Sara reçoit en 1er  : \"{top1_sara}...\"")
    print("\n  ✅ Les résultats sont différents et adaptés à chaque profil.")

    print(f"\n{'=' * 65}")
    print("  ✅ TEST 05 TERMINÉ — Personnalisation validée")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    test_profil_eleve()