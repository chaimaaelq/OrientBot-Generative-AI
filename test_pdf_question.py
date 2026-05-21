"""
TEST — Vérification réponses depuis les PDFs (toutes sources)
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from rag.vector_store import get_embedding_model, load_vector_store

emb = get_embedding_model()
vs  = load_vector_store(emb)

QUESTIONS_PDF = [
    # ── Répertoire établissements 2024-2025.pdf ──────────────────
    {
        "question": "ISTA conditions accès filières",
        "pdf": "Répertoire des établissements 2024-2025.pdf",
        "mots_cles": ["ISTA", "Bac", "titulaire"]
    },
    {
        "question": "CPGE classes préparatoires villes Maroc ingénieurs",
        "pdf": "Répertoire des établissements 2024-2025.pdf",
        "mots_cles": ["CPGE", "Préparatoires", "villes"]
    },
    {
        "question": "EST écoles supérieures technologie Agadir Fès",
        "pdf": "Répertoire des établissements 2024-2025.pdf",
        "mots_cles": ["EST", "Technologie", "Agadir"]
    },
    {
        "question": "faculté médecine pharmacie conditions bac concours",
        "pdf": "Répertoire des établissements 2024-2025.pdf",
        "mots_cles": ["médecine", "pharmacie", "Bac"]
    },
    {
        "question": "ISADAC institut arts dramatiques Rabat conditions",
        "pdf": "Répertoire des établissements 2024-2025.pdf",
        "mots_cles": ["dramatique", "Rabat", "Bac"]
    },

    # ── Liste universités privées 2025-2026.pdf ──────────────────
    {
        "question": "Université Internationale Casablanca privée autorisée",
        "pdf": "Liste des universités privées autorisées",
        "mots_cles": ["Internationale", "Casablanca", "privé"]
    },
    {
        "question": "UIR Université Internationale Rabat établissements",
        "pdf": "Liste des universités privées autorisées",
        "mots_cles": ["UIR", "Rabat", "Internationale"]
    },
    {
        "question": "Al Akhawayn université Ifrane privée autorisée",
        "pdf": "Liste des universités privées autorisées",
        "mots_cles": ["Akhawayn", "Ifrane"]
    },

    # ── Liste établissements privés 2025-2026.pdf ────────────────
   # Remplacer les 3 questions du dernier groupe par :
    {
        "question": "établissements privés autorisés numéro autorisation Maroc",
        "pdf": "Liste des établissements d'enseignement supérieur privé",
        "mots_cles": ["Privé", "autorisation", "établissement"]
    },
    {
        "question": "école supérieure privée Tanger Rabat Casablanca autorisée",
        "pdf": "Liste des établissements d'enseignement supérieur privé",
        "mots_cles": ["Privé", "Rabat", "Casablanca"]
    },
    {
        "question": "ESLSCA HEM ESTEM école privée Maroc reconnue",
        "pdf": "Liste des établissements d'enseignement supérieur privé",
        "mots_cles": ["ESLSCA", "Privé", "Rabat"]
    },
]

print("=" * 70)
print("TEST — RÉPONSES DEPUIS LES 3 PDFs")
print("=" * 70)

# Grouper par PDF
pdfs = {
    "Répertoire des établissements 2024-2025.pdf": [],
    "Liste des universités privées autorisées": [],
    "Liste des établissements d'enseignement supérieur privé": [],
}

resultats = []
for test in QUESTIONS_PDF:
    q    = test["question"]
    mots = test["mots_cles"]
    docs = vs.similarity_search(q, k=8)

    docs_pdf = [d for d in docs
                if d.metadata.get("type") == "pdf"
                and test["pdf"] in d.metadata.get("source", "")]
    contenu  = " ".join([d.page_content for d in docs])
    trouves  = [m for m in mots if m in contenu]
    score    = len(trouves) / len(mots)
    ok       = bool(docs_pdf) and score >= 0.6
    resultats.append(ok)

    statut = "✅" if ok else "❌"
    print(f"\n{statut} [{test['pdf'][:35]}...]")
    print(f"   Question          : {q}")
    print(f"   Docs PDF trouvés  : {len(docs_pdf)}/8")
    print(f"   Mots clés trouvés : {trouves}")
    print(f"   Score pertinence  : {score:.0%}")
    if docs_pdf:
        print(f"   Extrait           : {docs_pdf[0].page_content[:120]}...")

# Résumé par PDF
print("\n" + "=" * 70)
print("RÉSUMÉ PAR FICHIER PDF")
print("=" * 70)

groupes = {
    "Répertoire 2024-2025": QUESTIONS_PDF[:5],
    "Universités privées 2025-2026": QUESTIONS_PDF[5:8],
    "Établissements privés 2025-2026": QUESTIONS_PDF[8:],
}

for nom, tests in groupes.items():
    scores_g = []
    for i, test in enumerate(tests):
        idx = QUESTIONS_PDF.index(test)
        scores_g.append(resultats[idx])
    pct = sum(scores_g) / len(scores_g)
    barre = "█" * int(pct * 10) + "░" * (10 - int(pct * 10))
    statut = "✅" if pct >= 0.6 else "⚠️"
    print(f"{statut} {nom:<40} [{barre}] {sum(scores_g)}/{len(scores_g)}")

total = sum(resultats)
print(f"\n🎯 TOTAL : {total}/{len(QUESTIONS_PDF)} questions répondues depuis les PDFs")
if total >= 9:
    print("✅ Excellent — PDFs bien intégrés dans le RAG")
elif total >= 6:
    print("⚠️  Bon — PDFs partiellement utilisés")
else:
    print("❌ Problème — PDFs mal indexés")
print("=" * 70)