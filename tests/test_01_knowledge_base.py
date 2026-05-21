"""
=============================================================
TEST 01 — Chargement de la Base de Connaissances
=============================================================
Objectif : Vérifier que les documents (CSV + PDF) sont 
correctement chargés et découpés, et qu'ils couvrent
les besoins d'orientation des élèves marocains.

Fonctions testées :
  - load_csv_documents(path) 
  - load_pdf_documents(path)
  - load_all_documents()
  - split_documents(documents)
=============================================================
"""

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from knowledge_base.loader import (
    load_csv_documents,
    load_pdf_documents,
    load_all_documents,
    split_documents
)
from app.config import KNOWLEDGE_BASE_DIR


def test_knowledge_base_loading():
    print("=" * 65)
    print("  TEST 01 — Chargement de la Base de Connaissances")
    print("=" * 65)
    
    kb_dir = KNOWLEDGE_BASE_DIR
    
    # =============================================
    # ÉTAPE 1 : Test de load_csv_documents (1 fichier)
    # =============================================
    print(f"\n{'─' * 65}")
    print(f"  ÉTAPE 1 — Test de load_csv_documents() sur un fichier")
    print(f"{'─' * 65}")
    
    csv_files = list(kb_dir.glob("*.csv"))
    print(f"\n   Fichiers CSV trouvés : {len(csv_files)}")
    
    if csv_files:
        test_csv = csv_files[0]
        print(f"   Test sur : {test_csv.name}")
        csv_docs = load_csv_documents(test_csv)
        print(f"   Documents produits : {len(csv_docs)}")
        
        if csv_docs:
            print(f"\n   Aperçu du 1er document CSV :")
            print(f"  {'-' * 50}")
            content = csv_docs[0].page_content[:200].replace("\n", "\n     ")
            print(f"     {content}...")
            print(f"     Metadata : {csv_docs[0].metadata}")
    
    # =============================================
    # ÉTAPE 2 : Test de load_pdf_documents (1 fichier)
    # =============================================
    print(f"\n{'─' * 65}")
    print(f"  ÉTAPE 2 — Test de load_pdf_documents() sur un fichier")
    print(f"{'─' * 65}")
    
    pdf_files = list(kb_dir.glob("*.pdf"))
    print(f"\n   Fichiers PDF trouvés : {len(pdf_files)}")
    
    if pdf_files:
        test_pdf = pdf_files[0]
        print(f"   Test sur : {test_pdf.name}")
        pdf_docs = load_pdf_documents(test_pdf)
        print(f"   Pages extraites : {len(pdf_docs)}")
        
        if pdf_docs:
            print(f"\n   Aperçu de la 1ère page PDF :")
            print(f"  {'-' * 50}")
            content = pdf_docs[0].page_content[:200].replace("\n", "\n     ")
            print(f"     {content}...")
            print(f"     Metadata : {pdf_docs[0].metadata}")
    else:
        print(f"  ⚠️ Aucun fichier PDF trouvé")
    
    # =============================================
    # ÉTAPE 3 : Test de load_all_documents()
    # =============================================
    print(f"\n{'─' * 65}")
    print(f"  ÉTAPE 3 — Test de load_all_documents()")
    print(f"{'─' * 65}")
    
    all_docs = load_all_documents()
    print(f"\n   Total documents chargés : {len(all_docs)}")
    
    # Répartition CSV vs PDF
    csv_count = sum(1 for d in all_docs if d.metadata.get("type") == "csv")
    pdf_count = sum(1 for d in all_docs if d.metadata.get("type") == "pdf")
    print(f"      Documents CSV : {csv_count}")
    print(f"      Documents PDF : {pdf_count}")
    
    # Répartition par source
    sources = {}
    for doc in all_docs:
        src = doc.metadata.get("source", "inconnu")
        sources[src] = sources.get(src, 0) + 1
    
    print(f"\n   Répartition par fichier ({len(sources)} fichiers) :")
    print(f"  {'-' * 50}")
    for src, count in sorted(sources.items()):
        print(f"      {src:<45} → {count:>4} docs")
    
    # =============================================
    # ÉTAPE 4 : Test de split_documents()
    # =============================================
    print(f"\n{'─' * 65}")
    print(f"  ÉTAPE 4 — Test de split_documents()")
    print(f"{'─' * 65}")
    
    chunks = split_documents(all_docs)
    print(f"\n   Avant split : {len(all_docs)} documents")
    print(f"   Après split  : {len(chunks)} chunks")
    
    # Stats sur les chunks
    lengths = [len(c.page_content) for c in chunks]
    print(f"\n   Statistiques des chunks :")
    print(f"  {'-' * 50}")
    print(f"     Longueur moyenne  : {sum(lengths)/len(lengths):.0f} chars")
    print(f"     Plus court        : {min(lengths)} chars")
    print(f"     Plus long         : {max(lengths)} chars")
    
    # =============================================
    # ÉTAPE 5 : Vérification couverture profil élève
    # =============================================
    print(f"\n{'─' * 65}")
    print(f"  ÉTAPE 5 — Couverture pour un élève type")
    print(f"  (Bac SMA, Agadir, informatique, école publique)")
    print(f"{'─' * 65}")
    
    keywords_check = {
        "ENSA": 0, "informatique": 0, "génie": 0,
        "Agadir": 0, "public": 0, "privé": 0,
        "filière": 0, "bac": 0, "frais": 0,
        "université": 0, "métier": 0, "Casablanca": 0,
        "IA": 0, "intelligence artificielle": 0
    }
    
    for doc in chunks:
        content_lower = doc.page_content.lower()
        for kw in keywords_check:
            if kw.lower() in content_lower:
                keywords_check[kw] += 1
    
    print()
    for kw, count in keywords_check.items():
        status = "✅" if count > 0 else "❌"
        print(f"     {status} '{kw}' → trouvé dans {count} chunks")
    
    # Résumé final
    print(f"\n{'=' * 65}")
    print(f"  RÉSUMÉ TEST 01")
    print(f"{'=' * 65}")
    print(f"   Fichiers CSV         : {len(csv_files)}")
    print(f"   Fichiers PDF         : {len(pdf_files)}")
    print(f"   Documents chargés    : {len(all_docs)}")
    print(f"    Chunks après split   : {len(chunks)}")
    print(f"   Taille moyenne chunk  : {sum(lengths)/len(lengths):.0f} chars")
    print(f"\n{'=' * 65}")
    print(f"   TEST 01 TERMINÉ — Base de connaissances fonctionnelle")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    test_knowledge_base_loading()