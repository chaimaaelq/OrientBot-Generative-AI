"""
Chargeur de la base de connaissances (CSV + PDF)
"""
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import KNOWLEDGE_BASE_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_csv_documents(path: Path) -> List[Document]:
    docs = []
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception:
            df = pd.read_csv(path, encoding="cp1252")

    df.columns = [str(c).strip() for c in df.columns]

    # Détecter la colonne principale (nom établissement)
    col_principale = None
    for col in df.columns:
        col_ascii = col.lower()\
            .replace('é','e').replace('è','e').replace('ê','e')\
            .replace('à','a').replace('î','i').replace('û','u')
        if any(k in col_ascii for k in [
            'ecole', 'etablissement', 'institut',
            'universite', 'nom'
        ]) or 'instituts' in col_ascii:
            col_principale = col
            break

    # Détecter la colonne ville
    col_ville = None
    for col in df.columns:
        if 'ville' in col.lower() or 'localisation' in col.lower():
            col_ville = col
            break

    if col_principale:
        # Grouper par établissement + ville si disponible
        if col_ville:
            groupes = df.groupby([col_principale, col_ville], dropna=False)
        else:
            groupes = df.groupby([col_principale], dropna=False)

        for cle, groupe in groupes:
            # Construire le nom complet
            if col_ville and isinstance(cle, tuple):
                nom_etab, ville = cle
                content_parts = [
                    f"Établissement: {nom_etab}",
                    f"Ville: {ville}"
                ]
            else:
                nom_etab = cle
                content_parts = [f"Établissement: {nom_etab}"]

            # Ajouter toutes les autres colonnes
            for col in df.columns:
                if col in ([col_principale] + ([col_ville] if col_ville else [])):
                    continue
                valeurs = groupe[col].dropna().unique()
                valeurs = [
                    str(v).strip() for v in valeurs
                    if str(v).strip() and str(v).strip() != 'nan'
                ]
                if valeurs:
                    content_parts.append(f"{col}: {' | '.join(valeurs)}")

            content = "\n".join(content_parts)
            if content.strip():
                docs.append(Document(
                    page_content=content,
                    metadata={"source": path.name, "type": "csv"}
                ))
    else:
        # Une ligne = un document
        for _, row in df.iterrows():
            content_parts = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip() and str(val).strip() != 'nan':
                    content_parts.append(f"{col}: {str(val).strip()}")
            content = "\n".join(content_parts)
            if content.strip():
                docs.append(Document(
                    page_content=content,
                    metadata={"source": path.name, "type": "csv"}
                ))

    return docs


def extract_keywords(row, columns) -> str:
    """Extrait des mots-clés supplémentaires pour améliorer la recherche."""
    keywords = []
    col_lower = [str(c).lower() for c in columns]

    # Détecter le nom de l'établissement
    for i, col in enumerate(col_lower):
        if any(k in col for k in ['école', 'ecole', 'institut', 'université', 'etablissement', 'nom']):
            val = str(row[columns[i]]).strip()
            if val and val != 'nan':
                keywords.append(val)

    # Détecter la ville
    for i, col in enumerate(col_lower):
        if 'ville' in col or 'localisation' in col or 'région' in col:
            val = str(row[columns[i]]).strip()
            if val and val != 'nan':
                keywords.append(f"ville {val}")

    # Détecter le type
    for i, col in enumerate(col_lower):
        if 'type' in col:
            val = str(row[columns[i]]).strip()
            if val and val != 'nan':
                keywords.append(val)

    # Détecter les filières
    for i, col in enumerate(col_lower):
        if 'filière' in col or 'formation' in col or 'domaine' in col or 'spécialité' in col:
            val = str(row[columns[i]]).strip()
            if val and val != 'nan':
                keywords.append(val)

    return ", ".join(keywords)


def load_pdf_documents(path: Path) -> List[Document]:
    """Charge un fichier PDF page par page."""
    docs = []
    try:
        import PyPDF2
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": path.name, "page": i + 1, "type": "pdf"}
                    ))
    except Exception as e:
        print(f"⚠️ Erreur lecture PDF {path.name}: {e}")
    return docs


def load_all_documents() -> List[Document]:
    """Charge tous les documents CSV et PDF du dossier knowledge_base."""
    all_docs = []
    kb_dir = KNOWLEDGE_BASE_DIR

     # Fichiers à ignorer (mal formatés)
    FICHIERS_IGNORES = {
    #"universites-publiques-2024-2025.csv",
    #"metiers_maroc_par_filiere.csv",
    }

    # Fichiers à limiter (trop de lignes, dominent ChromaDB)
    FICHIERS_LIMITES = {
        "metiers_maroc_par_filiere.csv": 200,  # Max 200 docs
        "AI-based Career Recommendation System.csv": 200,
    }

    if not kb_dir.exists():
        print(f"⚠️ Dossier knowledge_base introuvable: {kb_dir}")
        return all_docs

    files = [f for f in kb_dir.iterdir() if f.is_file()]
    print(f"📂 {len(files)} fichiers trouvés dans la base de connaissances")

    for f in sorted(files):


        if f.name in FICHIERS_IGNORES:
            print(f"  ⏭️  Ignoré : {f.name}")
            continue
        
        if f.suffix.lower() == ".csv":
            print(f"  📊 Chargement CSV : {f.name}")
            docs = load_csv_documents(f)
            
            # Limiter les fichiers trop grands
            if f.name in FICHIERS_LIMITES:
                limite = FICHIERS_LIMITES[f.name]
                docs = docs[:limite]
                print(f"     → {len(docs)} documents (limité à {limite})")
            else:
                print(f"     → {len(docs)} documents")
            
            all_docs.extend(docs)
            
        elif f.suffix.lower() == ".pdf":
            print(f"  📄 Chargement PDF : {f.name}")
            docs = load_pdf_documents(f)
            all_docs.extend(docs)
            print(f"     → {len(docs)} pages")

    print(f"\n✅ Total : {len(all_docs)} documents chargés")

    return all_docs

def split_documents(documents: List[Document]) -> List[Document]:
    """Découpe les documents trop grands — garde les petits intacts."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    pdf_docs = [d for d in documents if d.metadata.get("type") == "pdf"]
    csv_docs = [d for d in documents if d.metadata.get("type") == "csv"]

    # Traiter les CSV
    csv_result = []
    for doc in csv_docs:
        if len(doc.page_content) > 1200:
            lines = doc.page_content.split('\n')
            header = '\n'.join(lines[:3])
            chunks = splitter.split_documents([doc])
            for chunk in chunks:
                if lines[0] not in chunk.page_content:
                    chunk.page_content = header + "\n---\n" + chunk.page_content
                csv_result.append(chunk)
        else:
            csv_result.append(doc)

    # Traiter les PDF
    pdf_chunks = splitter.split_documents(pdf_docs) if pdf_docs else []

    total = csv_result + pdf_chunks
    print(f"✂️  CSVs : {len(csv_result)} chunks | PDFs : {len(pdf_chunks)} chunks | Total : {len(total)}")
    return total