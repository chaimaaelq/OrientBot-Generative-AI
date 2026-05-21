"""
Connexion Mistral AI + Chaîne RAG — Version Optimisée PFE
Améliorations : query expansion, filtrage pertinent, contexte propre
"""
import re
from typing import List, Dict
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.config import (
    MISTRAL_API_KEY, MISTRAL_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS
)

# ═══════════════════════════════════════════════════════════════
# PROMPT SYSTÈME — optimisé pour fidélité et concision
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Tu es OrientBot, conseiller d'orientation scolaire expert du système éducatif marocain.

PROFIL DE L'ÉLÈVE :
{profil_context}

CONTEXTE (ta SEULE et UNIQUE source d'information) :
{context}

INSTRUCTIONS :
1. Réponds UNIQUEMENT avec les informations présentes dans le CONTEXTE ci-dessus
2. Pour chaque établissement mentionné, cite : Nom (Ville) | Diplôme | Filières pertinentes | Conditions d'accès
3. Si un détail n'est PAS dans le contexte → écris "Non précisé dans notre base"
4. Priorise les établissements de la ville de l'élève
5. Termine par UNE seule phrase de conseil

INTERDICTIONS ABSOLUES :
- N'invente AUCUN établissement, URL, site web, plateforme, nom de labo ou détail absent du contexte
- Pas de section "Autres options", "Si mobilité", "Parcours type", "Compétences à apprendre"
- Pas de mention de Coursera, Kaggle, MOOCs, certifications, outils à installer
- Pas de pourcentages de sélection inventés, pas de seuils non présents dans le contexte

Maximum 250 mots."""

# ═══════════════════════════════════════════════════════════════
# DICTIONNAIRES DE RÉFÉRENCE
# ═══════════════════════════════════════════════════════════════

VILLES = [
    "taroudant", "agadir", "rabat", "casablanca", "fès", "fes",
    "marrakech", "tanger", "oujda", "kenitra", "kénitra", "meknès",
    "meknes", "settat", "el jadida", "tetouan", "tétouan",
    "beni mellal", "safi", "khouribga", "ifrane", "benguerir",
    "ben guerir", "mohammadia", "salé", "sale", "larache",
    "nador", "errachidia", "ouarzazate", "laayoune", "dakhla",
    "al hoceima", "berrechid", "taza", "guelmim", "tiznit",
    "inezgane", "ait melloul", "temara", "khemisset"
]

ECOLES = [
    "ensa", "ensias", "emi", "encg", "iscae", "inpt", "ista",
    "ofppt", "fsjes", "fst", "fs", "fp", "flsh", "cpge", "um6p",
    "uir", "ensam", "ensaisd", "ena", "insea", "ita", "emsi",
    "hem", "aui", "al akhawayn", "mundiapolis", "uemf", "euromed",
    "faculte", "faculté", "université", "universite",
    "institut", "école", "ecole", "polydisciplinaire", "eniad",
    "epsinsia", "eeidia", "essia", "estem", "hestim"
]

SYNONYMES_NORMALISATION = {
    "branche": "filière", "branches": "filières",
    "spécialité": "filière", "spécialités": "filières",
    "specialite": "filière", "specialites": "filières",
    "option": "filière", "options": "filières",
    "section": "filière", "sections": "filières",
    "département": "filière", "departement": "filière",
    "cursus": "filière", "programme": "filière",
    "programmes": "filières", "formation": "filière",
    "formations": "filières", "domaine": "filière",
    "domaines": "filières", "parcours": "filière",
}

# ═══════════════════════════════════════════════════════════════
# QUERY EXPANSION — le changement le plus important
# Permet au retriever de trouver les bons documents
# ═══════════════════════════════════════════════════════════════

EXPANSION_TERMES = {
    # Domaines IA / Data
    "data scientist": "data science intelligence artificielle IA machine learning sciences des données big data statistiques",
    "data science": "intelligence artificielle IA machine learning sciences des données big data statistiques informatique",
    "intelligence artificielle": "IA data science machine learning deep learning sciences des données informatique",
    "ia": "intelligence artificielle data science machine learning sciences des données",
    "machine learning": "intelligence artificielle IA data science apprentissage automatique",
    "big data": "data science données massives informatique statistiques",

    # Informatique
    "informatique": "génie informatique développement logiciel programmation systèmes d'information réseaux",
    "génie informatique": "informatique ingénieur développement logiciel IA data",
    "développement": "informatique programmation génie logiciel",
    "cybersécurité": "sécurité informatique réseaux systèmes",

    # Médecine / Santé
    "médecine": "médecin docteur faculté médecine pharmacie santé dentaire",
    "pharmacie": "médecine santé faculté pharmacie",
    "dentaire": "médecine dentiste faculté",

    # Ingénierie
    "ingénieur": "école ingénieur ENSA ENSAM diplôme ingénieur génie",
    "génie civil": "ingénieur BTP construction travaux publics",
    "génie industriel": "ingénieur production industrielle",
    "génie électrique": "ingénieur électronique automatique électrotechnique",

    # Commerce / Gestion
    "commerce": "gestion management ENCG business économie",
    "gestion": "commerce management économie ENCG finance",
    "économie": "gestion commerce finance sciences économiques",
    "finance": "économie gestion comptabilité banque",

    # Autres
    "droit": "sciences juridiques avocat juriste faculté droit",
    "architecture": "architecte design urbanisme",

    # Diplômes
    "licence": "bac+3 licence fondamentale licence professionnelle",
    "master": "bac+5 master spécialisé master recherche",
    "cpge": "classes préparatoires prépa concours grandes écoles",
}

MOTS_VIDES = {
    "quels", "quelles", "quel", "quelle", "sont", "dans",
    "pour", "avec", "les", "des", "une", "qui", "comment",
    "existe", "existent", "trouver", "trouve", "est-ce",
    "après", "apres", "faire", "peut", "avoir", "être",
    "recommandez", "recommander", "proposent", "proposer",
    "disponibles", "disponible", "mieux", "meilleur",
    "vous", "nous", "mon", "mes", "votre",
}


def get_llm() -> ChatMistralAI:
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY manquante dans .env")
    return ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MISTRAL_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )


# ═══════════════════════════════════════════════════════════════
# NORMALISATION ET EXPANSION DE LA QUESTION
# ═══════════════════════════════════════════════════════════════

def normaliser_question(question: str) -> str:
    """Remplace les synonymes courants."""
    q = question
    for synonyme, remplacement in SYNONYMES_NORMALISATION.items():
        q = q.replace(synonyme, remplacement)
    return q


def expand_query(question: str) -> str:
    """Enrichit la question avec des termes associés pour meilleur retrieval.
    C'est le changement principal pour améliorer le rappel (17% → 50%+)."""
    question_lower = question.lower()
    expansions = []

    for terme, synonymes in EXPANSION_TERMES.items():
        if terme in question_lower:
            expansions.append(synonymes)

    if expansions:
        return question + " " + " ".join(expansions)
    return question


def enrich_with_profil(question: str, profil_context: str) -> str:
    """Enrichit la question avec les infos du profil élève."""
    if not profil_context:
        return question

    profil_lower = profil_context.lower()
    enrichments = []

    # Ajouter la ville
    for ville in VILLES:
        if ville in profil_lower and ville != "flexible":
            enrichments.append(ville)
            break

    # Ajouter le type d'établissement
    if "public" in profil_lower:
        enrichments.append("public")
    if "privé" in profil_lower or "prive" in profil_lower:
        enrichments.append("privé")

    # Ajouter les intérêts détectés
    interets_keywords = [
        "informatique", "intelligence artificielle", "data science",
        "médecine", "pharmacie", "commerce", "gestion", "droit",
        "ingénieur", "architecture", "agronomie", "mathématiques",
    ]
    for interet in interets_keywords:
        if interet in profil_lower:
            enrichments.append(interet)

    # Ajouter l'objectif de carrière
    objectifs = [
        "data scientist", "ingénieur", "médecin", "avocat",
        "architecte", "enseignant", "comptable", "pharmacien",
    ]
    for obj in objectifs:
        if obj in profil_lower:
            enrichments.append(obj)

    if enrichments:
        return question + " " + " ".join(enrichments)
    return question


def extraire_requetes_complementaires(question: str) -> list:
    """Génère des requêtes complémentaires ciblées (max 3)."""
    question_lower = question.lower()
    requetes = []

    # Requête par école spécifique mentionnée
    for ecole in ECOLES:
        if ecole in question_lower:
            requetes.append(f"{ecole} filières conditions accès")
            break  # une seule école à la fois

    # Requête par domaine mentionné
    for terme, synonymes in EXPANSION_TERMES.items():
        if terme in question_lower and len(requetes) < 3:
            requetes.append(f"établissements {terme} Maroc filières")
            break

    return requetes[:3]  # Max 3 requêtes complémentaires


# ═══════════════════════════════════════════════════════════════
# FILTRAGE ET TRI DES DOCUMENTS — plus strict
# ═══════════════════════════════════════════════════════════════

def scorer_document(doc: Document, question: str, profil_context: str) -> float:
    """Calcule un score de pertinence pour chaque document."""
    score = 0.0
    content_lower = doc.page_content.lower()
    question_lower = question.lower()
    profil_lower = profil_context.lower() if profil_context else ""

    # Score par mots-clés de la question
    mots_question = [
        m for m in re.findall(r'[a-zA-ZÀ-ÿ]+', question_lower)
        if len(m) > 3 and m not in MOTS_VIDES
    ]
    for mot in mots_question:
        if mot in content_lower:
            score += 2.0

    # Bonus ville du profil
    for ville in VILLES:
        if ville in profil_lower and ville in content_lower:
            score += 5.0
            break

    # Bonus type public/privé
    if "public" in profil_lower and any(k in content_lower for k in ["public", "publique"]):
        score += 3.0
    if ("privé" in profil_lower or "prive" in profil_lower) and any(k in content_lower for k in ["privé", "prive"]):
        score += 3.0

    # Bonus entités clés
    entites = [
        "ensa", "ensias", "emi", "encg", "eniad", "ensam", "ensaisd",
        "insea", "inpt", "cpge", "um6p", "emsi",
        "médecine", "pharmacie", "informatique", "ingénieur",
        "data science", "intelligence artificielle",
        "master", "licence", "filière",
    ]
    for entite in entites:
        if entite in question_lower and entite in content_lower:
            score += 4.0

    # Pénalité pour documents hors sujet
    if "informatique" in question_lower or "ia" in question_lower or "data" in question_lower:
        if any(k in content_lower for k in ["droit", "lettres", "sciences humaines", "flsh", "fsjes"]):
            if not any(k in content_lower for k in ["informatique", "data", "intelligence"]):
                score -= 10.0

    if "médecine" in question_lower:
        if any(k in content_lower for k in ["informatique", "génie", "ingénieur"]):
            if "médecine" not in content_lower:
                score -= 10.0

    return score


def filtrer_et_trier_docs(all_docs: list, question: str,
                          profil_context: str, max_docs: int = 8) -> list:
    """Filtre, score et trie les documents par pertinence."""
    # Dédupliquer
    seen = set()
    unique_docs = []
    for doc in all_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    # Scorer chaque document
    scored = [
        (doc, scorer_document(doc, question, profil_context))
        for doc in unique_docs
    ]

    # Retirer les docs avec score négatif (hors sujet)
    scored = [(doc, s) for doc, s in scored if s >= 0]

    # Trier par score décroissant
    scored.sort(key=lambda x: x[1], reverse=True)

    # Gérer public/privé
    profil_lower = profil_context.lower() if profil_context else ""
    if any(k in profil_lower for k in ["deux", "privé", "prive"]):
        docs_prives = [(d, s) for d, s in scored if any(k in d.page_content.lower() for k in ["privé", "prive"])]
        docs_publics = [(d, s) for d, s in scored if (d, s) not in docs_prives]
        # Réserver 2 slots pour les privés
        final = [d for d, s in docs_publics[:max_docs - 2]] + [d for d, s in docs_prives[:2]]
        # Compléter si nécessaire
        for d, s in scored:
            if d not in final and len(final) < max_docs:
                final.append(d)
        return final[:max_docs]

    return [doc for doc, s in scored[:max_docs]]


# ═══════════════════════════════════════════════════════════════
# CONSTRUCTION DE LA CHAÎNE RAG
# ═══════════════════════════════════════════════════════════════

def build_rag_chain(retriever, llm: ChatMistralAI,
                    profil_context: str, vectorstore=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

    reranker = None
    try:
        from rag.vector_store import get_reranker_retriever
        reranker = get_reranker_retriever(vectorstore)
        print("✅ Reranker Cross-Encoder activé")
    except Exception as e:
        print(f"⚠️ Reranker non disponible: {e}")

    return {
        "chain": chain,
        "retriever": retriever,
        "profil_context": profil_context,
        "vectorstore": vectorstore,
        "reranker": reranker
    }


# ═══════════════════════════════════════════════════════════════
# GÉNÉRATION DE RÉPONSE — pipeline nettoyé
# ═══════════════════════════════════════════════════════════════

def generate_response(rag_chain: dict, question: str,
                      chat_history: list = None) -> Dict:
    if chat_history is None:
        chat_history = []

    try:
        retriever      = rag_chain["retriever"]
        chain          = rag_chain["chain"]
        profil_context = rag_chain["profil_context"]
        vectorstore    = rag_chain.get("vectorstore")
        reranker       = rag_chain.get("reranker")

        # ── Étape 1 : Normaliser la question ──────────────────
        question_norm = normaliser_question(question)

        # ── Étape 2 : Expand query (CRITIQUE pour le rappel) ──
        question_expanded = expand_query(question_norm)
        question_expanded = enrich_with_profil(question_expanded, profil_context)

        # ── Étape 3 : Recherche principale ────────────────────
        if reranker:
            source_docs = reranker(question_expanded)
        elif hasattr(retriever, 'retrieve'):
            source_docs = retriever.retrieve(question_expanded)
        else:
            source_docs = retriever.invoke(question_expanded)

        # ── Étape 4 : Recherches complémentaires (max 3) ──────
        extra_docs = []
        if vectorstore:
            for req in extraire_requetes_complementaires(question_norm):
                try:
                    results = vectorstore.similarity_search(req, k=3)
                    extra_docs.extend(results)
                except Exception:
                    pass

        # ── Étape 5 : Recherche ville du profil ───────────────
        ville_docs = []
        if vectorstore:
            for ville in VILLES:
                if ville in profil_context.lower() and ville != "flexible":
                    try:
                        # Recherche ciblée ville + domaine d'intérêt
                        mots_question = [
                            m for m in re.findall(r'[a-zA-ZÀ-ÿ]+', question.lower())
                            if len(m) > 3 and m not in MOTS_VIDES
                        ]
                        domaine = " ".join(mots_question[:4])
                        ville_results = vectorstore.similarity_search(
                            f"{ville} {domaine} filières", k=5
                        )
                        ville_docs.extend(ville_results)
                    except Exception:
                        pass
                    break

        # ── Étape 6 : Injection privés si nécessaire ──────────
        prive_docs = []
        if any(k in profil_context.lower() for k in ["deux", "privé", "prive"]) and vectorstore:
            for ville in VILLES:
                if ville in profil_context.lower() and ville != "flexible":
                    try:
                        mots_q = [m for m in re.findall(r'[a-zA-ZÀ-ÿ]+', question.lower())
                                  if len(m) > 4 and m not in MOTS_VIDES]
                        domaine = " ".join(mots_q[:3])
                        prive_results = vectorstore.similarity_search(
                            f"établissement privé {ville} {domaine} frais scolarité", k=4
                        )
                        prive_docs.extend(prive_results)
                    except Exception:
                        pass
                    break

        # ── Étape 7 : Fusionner tous les documents ────────────
        all_docs = source_docs + extra_docs + ville_docs + prive_docs

        # ── Étape 8 : Filtrer et trier par pertinence ─────────
        docs_final = filtrer_et_trier_docs(
            all_docs, question_norm, profil_context, max_docs=8
        )

        if not docs_final:
            return {
                "answer": "Je n'ai pas trouvé d'informations pertinentes dans notre base.",
                "source_documents": [],
                "success": True
            }

        # ── Étape 9 : Construire le contexte ──────────────────
        context_parts = []
        for i, doc in enumerate(docs_final, 1):
            source = doc.metadata.get("source", "inconnu")
            context_parts.append(f"[Source {i} - {source}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_parts)

        # ── Étape 10 : Historique de conversation ─────────────
        messages_history = []
        for msg in chat_history[-4:]:
            if msg["role"] == "user":
                messages_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages_history.append(AIMessage(content=msg["content"]))

        # ── Étape 11 : Générer la réponse ─────────────────────
        answer = chain.invoke({
            "question": question_norm,
            "context": context,
            "profil_context": profil_context,
            "chat_history": messages_history,
        })

        return {
            "answer": answer,
            "source_documents": docs_final,
            "success": True
        }

    except Exception as e:
        return {
            "answer": f"Erreur : {str(e)}",
            "source_documents": [],
            "success": False
        }


# ═══════════════════════════════════════════════════════════════
# FONCTION DE TEST
# ═══════════════════════════════════════════════════════════════

from rag.vector_store import get_embedding_model, load_vector_store, get_retriever

def get_rag_chain():
    """Initialise le pipeline RAG complet."""
    embedding_model = get_embedding_model()
    vectorstore = load_vector_store(embedding_model)

    if vectorstore is None:
        raise ValueError("❌ ChromaDB introuvable ou vide.")

    retriever = get_retriever(vectorstore)
    llm = get_llm()

    return build_rag_chain(
        retriever=retriever,
        llm=llm,
        profil_context="Profil non spécifié",
        vectorstore=vectorstore
    )