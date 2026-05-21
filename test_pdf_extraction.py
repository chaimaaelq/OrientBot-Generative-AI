"""
TEST — Vérification extraction des PDFs
"""
import sys
from pathlib import Path
sys.path.insert(0, '.')

from knowledge_base.loader import load_pdf_documents

PDF_DIR = Path("knowledge_base/documents")
pdf_files = list(PDF_DIR.glob("*.pdf"))

print("=" * 60)
print(f"TEST EXTRACTION PDF — {len(pdf_files)} fichiers trouvés")
print("=" * 60)

for pdf in pdf_files:
    print(f"\n📄 {pdf.name}")
    print("-" * 40)
    docs = load_pdf_documents(pdf)
    print(f"   Pages extraites : {len(docs)}")
    
    if not docs:
        print("   ❌ Aucun contenu extrait !")
        continue
    
    # Afficher les 200 premiers caractères de chaque page
    for i, doc in enumerate(docs, 1):
        contenu = doc.page_content.strip()
        if contenu:
            print(f"   Page {i} ({len(contenu)} chars) :")
            print(f"   → {contenu[:200]}...")
        else:
            print(f"   Page {i} : ❌ VIDE")

print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
total_pages = 0
for pdf in pdf_files:
    docs = load_pdf_documents(pdf)
    vides = sum(1 for d in docs if not d.page_content.strip())
    total_pages += len(docs)
    statut = "✅" if len(docs) > 0 and vides == 0 else "⚠️"
    print(f"{statut} {pdf.name:<50} {len(docs)} pages | {vides} vides")

print(f"\nTotal : {total_pages} pages extraites")