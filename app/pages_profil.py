"""
OrientBot — Interface Streamlit principale
Page de saisie du profil élève
"""
import uuid
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    VILLES_MAROC, TYPES_BAC, DOMAINES_INTERET, METIERS_CIBLES
)
from backend.profil_eleve import (
    ProfilEleve, NiveauEtudes, TypeEtablissement, SituationSocioEco
)


# ── CSS personnalisé ──────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

    :root {
        --emerald:   #059669;
        --emerald-lt:#d1fae5;
        --sapphire:  #1d4ed8;
        --gold:      #d97706;
        --gold-lt:   #fef3c7;
        --ink:       #111827;
        --muted:     #6b7280;
        --surface:   #f9fafb;
        --card:      #ffffff;
        --border:    #e5e7eb;
        --radius:    14px;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--surface) !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Header hero */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c75 100%);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(5,150,105,0.25) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-family: 'Lora', serif;
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    .hero p {
        color: rgba(255,255,255,0.72);
        font-size: 1.05rem;
        margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: var(--emerald);
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 4px 12px;
        border-radius: 100px;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }

    /* Section cards */
    .section-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .section-title {
        font-weight: 700;
        font-size: 1rem;
        color: var(--ink);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid var(--emerald-lt);
    }

    /* Stickers / badges */
    .badge-public  { background:#dbeafe; color:#1d4ed8; border-radius:8px; padding:3px 10px; font-size:0.8rem; font-weight:600; }
    .badge-prive   { background:#fce7f3; color:#be185d; border-radius:8px; padding:3px 10px; font-size:0.8rem; font-weight:600; }
    .badge-neutre  { background:#f3f4f6; color:#374151; border-radius:8px; padding:3px 10px; font-size:0.8rem; font-weight:600; }

    /* Bouton principal */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, var(--emerald) 0%, #047857 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 15px rgba(5,150,105,0.3) !important;
        transition: all 0.2s !important;
        width: 100%;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(5,150,105,0.4) !important;
    }

    /* Inputs */
    .stSelectbox > div > div,
    .stMultiselect > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 10px !important;
        border-color: var(--border) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stSelectbox > div > div:focus-within,
    .stMultiselect > div > div:focus-within,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--emerald) !important;
        box-shadow: 0 0 0 3px rgba(5,150,105,0.15) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.85) !important;
    }

    /* Progress steps */
    .steps-bar {
        display: flex;
        gap: 8px;
        margin-bottom: 1.5rem;
    }
    .step {
        flex: 1;
        height: 4px;
        border-radius: 100px;
        background: var(--border);
    }
    .step.active { background: var(--emerald); }
    .step.done   { background: #a7f3d0; }

    /* Mention badge */
    .mention-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 100px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_mention(moy: float) -> tuple:
    if moy >= 16:   return "Très Bien", "#059669", "🏆"
    elif moy >= 14: return "Bien", "#2563eb", "⭐"
    elif moy >= 12: return "Assez Bien", "#d97706", "👍"
    elif moy >= 10: return "Passable", "#6b7280", "✅"
    else:            return "Non admis", "#dc2626", "❌"


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 2rem;'>
            <div style='font-size:2.5rem; margin-bottom:0.5rem;'>🧭</div>
            <div style='font-family:"Lora",serif; font-size:1.3rem;
                        font-weight:700; color:white;'>OrientBot</div>
            <div style='font-size:0.75rem; color:rgba(255,255,255,0.5);
                        letter-spacing:1px; text-transform:uppercase;'>
                Conseiller d'Orientation IA
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Navigation")
        page = st.radio("", [
            "🏠 Accueil",
            "👤 Mon Profil",
            "💬 Chatbot",
            "📊 Tableau de Bord",
        ], label_visibility="collapsed")

        st.markdown("---")
        

    return page


def render_hero():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🎓 Orientation Scolaire</div>
        <h1>Bienvenue sur OrientBot</h1>
        <p>Votre conseiller personnel basé sur l'IA générative pour trouver
           la meilleure filière et les meilleurs établissements au Maroc.</p>
    </div>
    """, unsafe_allow_html=True)


def render_profile_form() -> ProfilEleve:
    """Formulaire de saisie du profil — retourne un ProfilEleve."""

    st.markdown("## 👤 Mon Profil")
    st.markdown("Renseignez votre profil pour obtenir des recommandations personnalisées.")

    # ── Étape 1 : Infos générales ────────────────────────────────────────────
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📋 Informations Générales</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            prenom = st.text_input("Prénom *", placeholder="Ex : Yasmine", key="prenom")
        with col2:
            niveau_label = st.selectbox(
                "Niveau scolaire actuel *",
                [n.value for n in NiveauEtudes],
                key="niveau"
            )

    niveau = NiveauEtudes(niveau_label)
    est_bachelier = niveau in [NiveauEtudes.BACHELIER, NiveauEtudes.ETUDIANT]
    est_secondaire = niveau in [NiveauEtudes.DEUXIEME_BAC, NiveauEtudes.PREMIERE_BAC]

    # ── Étape 2 : Infos Bac (si bachelier) ──────────────────────────────────
    type_bac    = None
    moyenne_bac = None
    mention_bac = None
    note_maths  = note_physique = note_svt = note_francais = None
    note_anglais = note_philo = note_arabe = note_eco = None

    if est_bachelier or est_secondaire:
        st.markdown("""
        <div class="section-card" style="margin-top:1rem;">
            <div class="section-title">🎓 Informations sur le Baccalauréat</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            type_bac = st.selectbox("Type de Baccalauréat *", TYPES_BAC, key="type_bac")
        with col2:
            if est_bachelier:
                moyenne_bac = st.number_input(
                    "Moyenne générale /20 *",
                    min_value=0.0, max_value=20.0,
                    value=12.0, step=0.25,
                    key="moyenne_bac"
                )
                if moyenne_bac > 0:
                    mention, color, emoji = get_mention(moyenne_bac)
                    mention_bac = mention
                    st.markdown(f"""
                    <div class="mention-badge" style="background:{color}22; color:{color};">
                        {emoji} Mention : {mention}
                    </div>
                    """, unsafe_allow_html=True)

        # Notes par matière
        st.markdown("**📝 Notes par matière** *(optionnel — améliore les recommandations)*")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            maths_val = st.number_input("Mathématiques", 0.0, 20.0, step=0.25, key="n_maths", format="%.2f")
            note_maths = maths_val if maths_val > 0 else None
            phys_val = st.number_input("Physique-Chimie", 0.0, 20.0, step=0.25, key="n_phys", format="%.2f")
            note_physique = phys_val if phys_val > 0 else None
        with col2:
            svt_val = st.number_input("SVT / Bio", 0.0, 20.0, step=0.25, key="n_svt", format="%.2f")
            note_svt = svt_val if svt_val > 0 else None
            fr_val = st.number_input("Français", 0.0, 20.0, step=0.25, key="n_fr", format="%.2f")
            note_francais = fr_val if fr_val > 0 else None
        with col3:
            en_val = st.number_input("Anglais", 0.0, 20.0, step=0.25, key="n_en", format="%.2f")
            note_anglais = en_val if en_val > 0 else None
            philo_val = st.number_input("Philosophie", 0.0, 20.0, step=0.25, key="n_philo", format="%.2f")
            note_philo = philo_val if philo_val > 0 else None
        with col4:
            arabe_val = st.number_input("Arabe", 0.0, 20.0, step=0.25, key="n_arabe", format="%.2f")
            note_arabe = arabe_val if arabe_val > 0 else None
            eco_val = st.number_input("Économie", 0.0, 20.0, step=0.25, key="n_eco", format="%.2f")
            note_eco = eco_val if eco_val > 0 else None

    # ── Étape 3 : Intérêts & Objectifs ──────────────────────────────────────
    st.markdown("""
    <div class="section-card" style="margin-top:1rem;">
        <div class="section-title">🌟 Centres d'Intérêt & Objectifs</div>
    </div>
    """, unsafe_allow_html=True)

    interets = st.multiselect(
        "Domaines qui vous intéressent *",
        DOMAINES_INTERET,
        max_selections=5,
        key="interets",
        help="Choisissez jusqu'à 5 domaines"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        objectif = st.selectbox("Métier ou domaine cible", ["— Pas encore décidé —"] + METIERS_CIBLES, key="objectif")
        objectif_carriere = None if objectif == "— Pas encore décidé —" else objectif
    with col2:
        description = st.text_area(
            "Décrivez vos ambitions (optionnel)",
            placeholder="Ex : Je veux devenir ingénieur en IA, j'aime les maths et l'informatique...",
            height=100,
            key="description"
        )

    # ── Étape 4 : Préférences & Socio-économique ─────────────────────────────
    st.markdown("""
    <div class="section-card" style="margin-top:1rem;">
        <div class="section-title">🏫 Préférences d'Établissement</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ville = st.selectbox("🏙️ Ville souhaitée", VILLES_MAROC, key="ville")
        ville_preferee = None if ville == "Toutes" else ville
    with col2:
        type_etab_label = st.selectbox(
            "🏛️ Type d'établissement",
            [t.value for t in TypeEtablissement],
            key="type_etab"
        )
        type_etab = TypeEtablissement(type_etab_label)
    with col3:
        mobilite = st.checkbox("📍 Je suis prêt(e) à quitter ma ville", value=True, key="mobilite")

    col1, col2 = st.columns(2)
    with col1:
        socio_label = st.selectbox(
            "💰 Situation socio-économique",
            [s.value for s in SituationSocioEco],
            key="socio",
            help="Aide à adapter les recommandations au budget"
        )
        situation_socio = SituationSocioEco(socio_label)
    with col2:
        budget_input = st.number_input(
            "💵 Budget max /an (MAD) — optionnel",
            min_value=0, max_value=300000,
            value=0, step=5000,
            key="budget"
        )
        budget = budget_input if budget_input > 0 else None

    # ── Validation ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    valid = bool(prenom and niveau_label and interets)
    if not valid:
        st.info("ℹ️ Remplissez au minimum : prénom, niveau et domaines d'intérêt.")

    submitted = st.button("🚀 Créer mon profil et démarrer", type="primary", disabled=not valid)

    if submitted and valid:
        profil = ProfilEleve(
            prenom=prenom,
            niveau=niveau,
            type_bac=type_bac,
            moyenne_bac=moyenne_bac if moyenne_bac and moyenne_bac > 0 else None,
            mention_bac=mention_bac,
            note_maths=note_maths,
            note_physique=note_physique,
            note_svt=note_svt,
            note_francais=note_francais,
            note_anglais=note_anglais,
            note_philosophie=note_philo,
            note_arabe=note_arabe,
            note_economie=note_eco,
            interets=interets,
            objectif_carriere=objectif_carriere,
            description_objectif=description or None,
            ville_preferee=ville_preferee,
            type_etablissement=type_etab,
            situation_socio_eco=situation_socio,
            budget_annuel_max=budget,
            mobilite=mobilite,
        )
        return profil

    return None
