"""
=============================================================
TEST 07 — Évaluation RAGAS / Métriques personnalisées
=============================================================
Objectif : Évaluer la qualité des réponses du système
sur des questions liées au profil de l'apprenant.
=============================================================
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.ragas_eval import evaluate_response
from rag.llm import get_rag_chain, generate_response
from backend.profil_eleve import ProfilEleve


def test_ragas_evaluation():
    print("=" * 65)
    print("  TEST 07 — Évaluation RAGAS / Métriques personnalisées")
    print("=" * 65)

    test_cases = [
        {
            "profil": ProfilEleve(
                prenom="Ahmed",
                niveau="Bachelier (Bac obtenu)",
                type_bac="Sciences Mathématiques A",
                moyenne_bac=15.8,
                note_maths=17,
                note_physique=15,
                interets=["informatique", "intelligence artificielle", "data science"],
                objectif_carriere="Ingénieur en intelligence artificielle",
                ville_preferee="Agadir",
                type_etablissement="Public",
                situation_socio_eco="Modeste (budget limité)",
                budget_annuel_max=None,
                mobilite=True
            ),
            "question": "Selon mon profil, mes résultats en mathématiques et mon intérêt pour l'IA, quelles filières publiques me conviennent ?"
        },
        {
            "profil": ProfilEleve(
                prenom="Fatima",
                niveau="Bachelier (Bac obtenu)",
                type_bac="Sciences de la Vie et de la Terre",
                moyenne_bac=14.2,
                note_svt=16,
                note_physique=13,
                interets=["médecine", "pharmacie", "biologie"],
                objectif_carriere="Médecin ou pharmacienne",
                ville_preferee="Casablanca",
                type_etablissement="Privé",
                situation_socio_eco="Aisé",
                budget_annuel_max=70000,
                mobilite=False
            ),
            "question": "Selon mon profil, mon bac SVT, mon budget et mon intérêt pour la santé, quelles options privées sont adaptées ?"
        },
        {
            "profil": ProfilEleve(
                prenom="Youssef",
                niveau="Bachelier (Bac obtenu)",
                type_bac="Sciences Mathématiques A",
                moyenne_bac=13.5,
                note_maths=14,
                note_anglais=16,
                interets=["data science", "statistiques", "intelligence artificielle"],
                objectif_carriere="Data scientist",
                ville_preferee="Fès",
                type_etablissement="Public",
                situation_socio_eco="Modeste (budget limité)",
                budget_annuel_max=None,
                mobilite=True
            ),
            "question": "Selon mon profil scolaire et mon intérêt pour la data science, quelles formations publiques peuvent m'aider ?"
        }
    ]

    print("\n  ⏳ Initialisation du pipeline RAG...")
    rag_chain = get_rag_chain()
    print("  ✅ Pipeline prêt")

    all_results = []

    for i, tc in enumerate(test_cases):
        profil = tc["profil"]
        question = tc["question"]

        print(f"\n{'=' * 65}")
        print(f"  Évaluation {i + 1}/{len(test_cases)} — {profil.prenom}")
        print(f"{'=' * 65}")

        print(f"  👤 Profil : {profil.prenom}")
        print(f"     Bac      : {profil.type_bac}")
        print(f"     Moyenne  : {profil.moyenne_bac}/20")
        print(f"     Ville    : {profil.ville_preferee}")
        print(f"     Type     : {profil.type_etablissement}")
        print(f"     Intérêts : {', '.join(profil.interets)}")

        print(f"\n  📝 Question : {question}")

        start = time.time()

        rag_chain["profil_context"] = profil.to_context_string()

        result = generate_response(
            rag_chain=rag_chain,
            question=question,
            chat_history=[]
        )

        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        metrics = evaluate_response(
            question=question,
            answer=answer,
            source_docs=source_docs
        )

        eval_time = time.time() - start

        print("\n  💬 Réponse générée :")
        print("  " + "-" * 55)
        for line in answer.split("\n"):
            print(f"    {line}")

        print("\n  📊 Métriques :")
        print(f"  {'─' * 45}")
        print(f"  │ Faithfulness       : {metrics.get('faithfulness', 0) * 100:>6.1f}%")
        print(f"  │ Answer Relevancy   : {metrics.get('answer_relevancy', 0) * 100:>6.1f}%")
        print(f"  │ Context Precision  : {metrics.get('context_precision', 0) * 100:>6.1f}%")
        print(f"  │ Context Recall     : {metrics.get('context_recall', 0) * 100:>6.1f}%")
        print(f"  │ Answer Correctness : {metrics.get('answer_correctness', 0) * 100:>6.1f}%")
        print(f"  │ Overall Score      : {metrics.get('overall_score', 0) * 100:>6.1f}%")
        print(f"  {'─' * 45}")
        print(f"  ⏱ Temps : {eval_time:.2f}s")

        all_results.append({
            "question": question[:45] + "...",
            "profil": profil.prenom,
            **{k: v * 100 for k, v in metrics.items()}
        })

    print(f"\n\n{'=' * 95}")
    print("  TABLEAU RÉCAPITULATIF — ÉVALUATION DU RAG")
    print(f"{'=' * 95}")

    print(f"\n  {'Profil':<10} {'Question':<45} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Recall':>6} {'Corr':>6} {'Global':>6}")
    print(f"  {'─' * 92}")

    keys = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "overall_score"
    ]

    totals = {k: 0 for k in keys}

    for r in all_results:
        print(
            f"  {r['profil']:<10} "
            f"{r['question']:<45} "
            f"{r.get('faithfulness', 0):>5.1f}% "
            f"{r.get('answer_relevancy', 0):>5.1f}% "
            f"{r.get('context_precision', 0):>5.1f}% "
            f"{r.get('context_recall', 0):>5.1f}% "
            f"{r.get('answer_correctness', 0):>5.1f}% "
            f"{r.get('overall_score', 0):>5.1f}%"
        )

        for k in keys:
            totals[k] += r.get(k, 0)

    n = len(all_results)

    print(f"  {'─' * 92}")
    print(
        f"  {'MOYENNE':<10} "
        f"{'':<45} "
        f"{totals['faithfulness'] / n:>5.1f}% "
        f"{totals['answer_relevancy'] / n:>5.1f}% "
        f"{totals['context_precision'] / n:>5.1f}% "
        f"{totals['context_recall'] / n:>5.1f}% "
        f"{totals['answer_correctness'] / n:>5.1f}% "
        f"{totals['overall_score'] / n:>5.1f}%"
    )

    print(f"\n{'=' * 65}")
    print("  ✅ TEST 07 TERMINÉ — Évaluation RAG complète")
    print(f"{'=' * 65}")

if __name__ == "__main__":
    test_ragas_evaluation()