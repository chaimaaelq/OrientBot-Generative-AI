"""
Configuration centrale du projet Chatbot d'Orientation Scolaire
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = "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"
# ── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base" / "documents"
CHROMA_DIR = DATA_DIR / "chroma_db"
SQLITE_PATH = DATA_DIR / "orientation.db"

# Crée les dossiers si besoin
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM (Mistral) ────────────────────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL   = "mistral-medium-latest"   # ou "open-mistral-7b" pour la version gratuite
LLM_TEMPERATURE = 0.03
LLM_MAX_TOKENS  = 800

# ── Embeddings ────────────────────────────────────────────────────────────────
#EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K_RETRIEVAL = 8

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "orientation_maroc"

# ── Mémoire conversationnelle ─────────────────────────────────────────────────
MEMORY_MAX_TOKEN_LIMIT = 2000

# ── Évaluation RAGAS ─────────────────────────────────────────────────────────
RAGAS_SAMPLE_SIZE = 10

# ── Villes marocaines disponibles ─────────────────────────────────────────────
VILLES_MAROC = [
    "Toutes", "Taroudant","Casablanca", "Rabat", "Marrakech", "Fès", "Tanger",
    "Agadir", "Meknès", "Oujda", "Kénitra", "Tétouan", "Salé",
    "Mohammedia", "Khouribga", "Béni Mellal", "El Jadida", "Settat",
    "Nador", "Larache", "Khémisset", "Guelmim"
]

# ── Types de bac marocain ─────────────────────────────────────────────────────
TYPES_BAC = [
    "Sciences Mathématiques A",
    "Sciences Mathématiques B",
    "Sciences Physiques",
    "Sciences de la Vie et de la Terre (SVT)",
    "Sciences Économiques et Gestion",
    "Lettres et Sciences Humaines (LSH)",
    "Bac Technique",
    "Bac Professionnel",
]

# ── Domaines d'intérêt ────────────────────────────────────────────────────────
DOMAINES_INTERET = [
    "Intelligence Artificielle & Data Science",
    "Informatique & Développement",
    "Médecine & Santé",
    "Ingénierie & Génie Civil",
    "Commerce & Gestion",
    "Droit & Sciences Juridiques",
    "Architecture & Design",
    "Éducation & Enseignement",
    "Arts & Communication",
    "Mathématiques & Physique",
    "Biologie & Chimie",
    "Finance & Économie",
    "Entrepreneuriat & Business",
    "Sciences Politiques",
    "Agronomie & Environnement",
]

# ── Métiers cibles ─────────────────────────────────────────────────────────────
METIERS_CIBLES = [
    "Ingénieur Informatique",
    "Data Scientist / AI Engineer",
    "Médecin / Dentiste",
    "Avocat / Juriste",
    "Architecte",
    "Enseignant / Chercheur",
    "Chef d'entreprise / Entrepreneur",
    "Comptable / Auditeur",
    "Ingénieur Civil / BTP",
    "Designer / Graphiste",
    "Journaliste / Communicant",
    "Pharmacien",
    "Économiste / Analyste",
    "Autre",
]
