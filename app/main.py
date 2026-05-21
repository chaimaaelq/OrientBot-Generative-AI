"""
OrientBot — Point d'entrée principal Streamlit
Lancer avec : streamlit run app/main.py
"""
import uuid
import sys
from pathlib import Path

import streamlit as st

# Ajout du répertoire racine au path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="OrientBot — Conseiller d'Orientation IA",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.pages_profil  import inject_css, render_sidebar, render_hero, render_profile_form
from app.pages_chat    import render_chat_interface, render_profil_summary
from app.pages_dashboard import render_dashboard
from backend.database  import DatabaseManager


# ── Initialisation ────────────────────────────────────────────────────────────
inject_css()
db = DatabaseManager()

# ── Session State ─────────────────────────────────────────────────────────────
if "session_id"    not in st.session_state: st.session_state.session_id    = str(uuid.uuid4())
if "profil"        not in st.session_state: st.session_state.profil        = None
if "chain"         not in st.session_state: st.session_state.chain         = None
if "rag_ready"     not in st.session_state: st.session_state.rag_ready     = False
if "show_metrics"  not in st.session_state: st.session_state.show_metrics  = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
page = render_sidebar()

# Afficher résumé profil dans sidebar si disponible
if st.session_state.profil:
    with st.sidebar:
        st.markdown("---")
        render_profil_summary(st.session_state.profil)
        st.session_state.show_metrics = st.checkbox(
            "Afficher métriques RAGAS", value=st.session_state.show_metrics
        )
        if st.button("🔄 Nouveau profil", use_container_width=True):
            st.session_state.profil        = None
            st.session_state.chain         = None
            st.session_state.rag_ready     = False
            st.session_state.messages      = []
            st.session_state.session_id    = str(uuid.uuid4())
            st.rerun()


# ── Router principal ──────────────────────────────────────────────────────────
if page == "🏠 Accueil":
    render_hero()

    col1, col2, col3 = st.columns(3)
    card_style = """
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        height: 180px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    """
    with col1:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:2rem; margin-bottom:0.7rem;">🎯</div>
            <div style="font-weight:700; font-size:1rem; color:#111827; margin-bottom:0.4rem;">
                Recommandations Personnalisées
            </div>
            <div style="font-size:0.85rem; color:#6b7280;">
                Filières, établissements et parcours adaptés à votre profil, notes et budget.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:2rem; margin-bottom:0.7rem;">🏛️</div>
            <div style="font-weight:700; font-size:1rem; color:#111827; margin-bottom:0.4rem;">
                Base de Connaissances Maroc
            </div>
            <div style="font-size:0.85rem; color:#6b7280;">
                Données officielles 2024–2025 : établissements, seuils, conditions d'accès.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="{card_style}">
            <div style="font-size:2rem; margin-bottom:0.7rem;">🤖</div>
            <div style="font-weight:700; font-size:1rem; color:#111827; margin-bottom:0.4rem;">
                IA Générative RAG
            </div>
            <div style="font-size:0.85rem; color:#6b7280;">
                Architecture Mistral AI + ChromaDB pour des réponses précises et sourcées.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Section d'initialisation du RAG
    with st.expander("⚙️ Configuration du système (Admin)", expanded=False):
        st.markdown("**Indexation de la base de connaissances**")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(
                "Avant d'utiliser le chatbot, indexez vos fichiers CSV et PDF dans ChromaDB.\n\n"
                "Placez vos fichiers dans `knowledge_base/documents/` puis cliquez sur Indexer."
            )
        with col2:
            if st.button("📚 Indexer la base", type="primary"):
                with st.spinner("Indexation en cours..."):
                    try:
                        from knowledge_base.indexer import index_knowledge_base
                        vs = index_knowledge_base()
                        if vs:
                            st.session_state.vectorstore = vs
                            st.success("✅ Base indexée !")
                        else:
                            st.warning("⚠️ Aucun document trouvé.")
                    except Exception as e:
                        st.error(f"Erreur : {e}")

            if st.button("🔗 Initialiser le chatbot"):
                with st.spinner("Chargement du LLM et du retriever..."):
                    try:
                        from rag.vector_store import get_embedding_model, load_vector_store, get_retriever
                        from rag.llm import get_llm, build_rag_chain

                        emb   = get_embedding_model()
                        vs    = load_vector_store(emb)
                        if vs is None:
                            st.error("❌ Base non indexée. Indexez d'abord.")
                        else:
                            from rag.vector_store import get_embedding_model, load_vector_store, get_retriever, build_hybrid_retriever
                            from rag.llm import get_llm, build_rag_chain
                            from langchain_core.documents import Document

                            retr = get_retriever(vs)
                            llm  = get_llm()
                            profil_ctx = (
                                st.session_state.profil.to_context_string()
                                if st.session_state.profil else "Profil non défini."
                            )

                            # Construire hybrid retriever
                            all_data = vs.get()
                            all_docs = [
                                Document(page_content=doc, metadata=meta)
                                for doc, meta in zip(all_data['documents'], all_data['metadatas'])
                            ]
                            hybrid = build_hybrid_retriever(vs, all_docs)
                            final_retr = hybrid if hybrid else retr

                            chain = build_rag_chain(final_retr, llm, profil_ctx, vs)
                            st.session_state.chain     = chain
                            st.session_state.rag_ready = True
                            st.success("✅ Chatbot prêt !")
                    except Exception as e:
                        st.error(f"Erreur LLM : {e}")

    # Bouton démarrer
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("👤 Créer mon profil et commencer", type="primary"):
        st.session_state._nav_target = "👤 Mon Profil"
        st.rerun()

elif page == "👤 Mon Profil":
    profil = render_profile_form()
    if profil:
        st.session_state.profil = profil
        db.save_session(st.session_state.session_id, profil.model_dump())
        st.success(f"✅ Profil de **{profil.prenom}** enregistré ! Rendez-vous dans l'onglet 💬 Chatbot.")
        st.balloons()

        # Initialiser le RAG si disponible
        if not st.session_state.rag_ready:
            with st.spinner("Chargement du système RAG..."):
                try:
                    from rag.vector_store import get_embedding_model, load_vector_store, get_retriever
                    from rag.llm import get_llm, build_rag_chain
                    emb  = get_embedding_model()
                    vs   = load_vector_store(emb)
                    if vs:
                        from rag.vector_store import get_embedding_model, load_vector_store, get_retriever, build_hybrid_retriever
                        from rag.llm import get_llm, build_rag_chain
                        from langchain_core.documents import Document

                        retr = get_retriever(vs)
                        llm  = get_llm()
                        all_data = vs.get()
                        all_docs = [
                            Document(page_content=doc, metadata=meta)
                            for doc, meta in zip(all_data['documents'], all_data['metadatas'])
                        ]
                        hybrid = build_hybrid_retriever(vs, all_docs)
                        final_retr = hybrid if hybrid else retr
                        chain = build_rag_chain(final_retr, llm, profil.to_context_string(), vs)
                        st.session_state.chain     = chain
                        st.session_state.rag_ready = True
                except Exception:
                    pass  # Mode démo si pas de clé API

elif page == "💬 Chatbot":
    if st.session_state.profil is None:
        st.warning("⚠️ Veuillez d'abord remplir votre profil dans l'onglet **👤 Mon Profil**.")
        if st.button("Aller au profil →"):
            st.rerun()
    else:
        if not st.session_state.rag_ready:
            st.info(
                "ℹ️ Le système RAG n'est pas encore initialisé. "
                "Le chatbot fonctionnera en mode démonstration.\n\n"
                "Pour activer le RAG : Accueil → Configuration → Indexer + Initialiser."
            )
        render_chat_interface(
            chain=st.session_state.chain,
            db=db,
            session_id=st.session_state.session_id,
            profil=st.session_state.profil
        )

elif page == "📊 Tableau de Bord":
    render_dashboard(db)
