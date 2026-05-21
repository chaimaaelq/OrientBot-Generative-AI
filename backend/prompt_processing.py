"""
Traitement et enrichissement des prompts utilisateur
"""
import re
from typing import Optional
from backend.profil_eleve import ProfilEleve


def clean_text(text: str) -> str:
    """Nettoie et normalise le texte."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def enrich_question_with_profile(question: str, profil: ProfilEleve) -> str:
    """
    Enrichit la question avec le contexte du profil élève
    pour améliorer la qualité du retrieval.
    """
    # Extrait les mots-clés pertinents du profil
    keywords = []

    if profil.type_bac:
        keywords.append(profil.type_bac)
    if profil.ville_preferee:
        keywords.append(profil.ville_preferee)
    if profil.type_etablissement and profil.type_etablissement != "Les deux (pas de préférence)":
        keywords.append(profil.type_etablissement)
    if profil.interets:
        keywords.extend(profil.interets[:2])
    if profil.objectif_carriere and profil.objectif_carriere != "Autre":
        keywords.append(profil.objectif_carriere)

    # Construit la question enrichie
    if keywords:
        enriched = f"{question} [Contexte: {', '.join(keywords)}]"
    else:
        enriched = question

    return clean_text(enriched)


def detect_question_intent(question: str) -> str:
    """
    Détecte l'intention principale de la question.
    Retourne : 'filière', 'établissement', 'métier', 'condition', 'budget', 'général'
    """
    question_lower = question.lower()

    if any(w in question_lower for w in ['filière', 'formation', 'études', 'spécialité', 'cursus', 'parcours']):
        return 'filière'
    elif any(w in question_lower for w in ['école', 'université', 'établissement', 'faculté', 'institut']):
        return 'établissement'
    elif any(w in question_lower for w in ['métier', 'carrière', 'profession', 'emploi', 'travail', 'job']):
        return 'métier'
    elif any(w in question_lower for w in ['condition', 'admission', 'seuil', 'concours', 'dossier', 'prérequis']):
        return 'condition'
    elif any(w in question_lower for w in ['prix', 'frais', 'coût', 'budget', 'bourse', 'gratuit', 'payant']):
        return 'budget'
    else:
        return 'général'


def format_sources_summary(source_docs: list) -> str:
    """Formate un résumé des sources utilisées."""
    if not source_docs:
        return ""
    sources = set()
    for doc in source_docs:
        src = doc.metadata.get("source", "inconnue")
        sources.add(src)
    return f"📚 Sources : {', '.join(sorted(sources))}"
