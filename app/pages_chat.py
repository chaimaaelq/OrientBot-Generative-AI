"""
Page Chatbot — interface de conversation avec l'IA
"""
import streamlit as st
from typing import Optional
from backend.profil_eleve import ProfilEleve
from evaluation.ragas_eval import evaluate_response, format_metrics_display


def render_profil_summary(profil: ProfilEleve):
    """Affiche un résumé compact du profil dans la sidebar."""
    mention = profil.mention_bac or ""
    budget_str = f"{profil.budget_annuel_max:,} MAD/an" if profil.budget_annuel_max else "Non défini"
    ville = profil.ville_preferee or "Flexible"

    st.markdown(f"""
    <div style="background:rgba(5,150,105,0.1); border:1px solid rgba(5,150,105,0.3);
                border-radius:12px; padding:1rem; margin-bottom:1rem;">
        <div style="font-weight:700; font-size:1rem; margin-bottom:0.5rem; color:#059669;">
            👤 {profil.prenom}
        </div>
        <div style="font-size:0.82rem; color:rgba(255,255,255,0.75); line-height:1.8;">
            📚 {profil.niveau}<br>
            {"🎓 " + profil.type_bac + "<br>" if profil.type_bac else ""}
            {"⭐ Moy: " + str(profil.moyenne_bac) + "/20 " + mention + "<br>" if profil.moyenne_bac else ""}
            🌟 {", ".join(profil.interets[:2])}{"..." if len(profil.interets) > 2 else ""}<br>
            🏙️ {ville} | {profil.type_etablissement}<br>
            💰 {budget_str}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_interface(chain, db, session_id: str, profil: ProfilEleve):
    """Interface principale de chat."""

    # ── Messages d'exemple ────────────────────────────────────────────────────
    QUESTIONS_SUGGEREES = [
        "Quelles filières correspondent à mon profil et mes notes ?",
        "Quels établissements sont disponibles dans ma ville ?",
        "Quels sont les seuils d'admission pour les écoles d'ingénieurs ?",
        "Comparez les options publiques et privées pour moi",
        "Quels métiers correspondent à mes intérêts ?",
    ]

    # ── Historique ────────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "show_metrics" not in st.session_state:
        st.session_state.show_metrics = False

    # Message de bienvenue
    if not st.session_state.messages:
        welcome = f"""Bonjour **{profil.prenom}** ! 👋

Je suis **OrientBot**, votre conseiller d'orientation personnalisé. J'ai bien pris en compte votre profil :

- 📚 Niveau : **{profil.niveau}**
{f"- 🎓 Bac : **{profil.type_bac}** | Moyenne : **{profil.moyenne_bac}/20**" if profil.moyenne_bac else ""}
- 🌟 Intérêts : **{", ".join(profil.interets)}**
- 🏙️ Ville souhaitée : **{profil.ville_preferee or "Flexible"}**
- 🏛️ Préférence : **{profil.type_etablissement}**

Je suis prêt à vous aider ! Posez-moi vos questions sur les filières, établissements et parcours. """

        st.session_state.messages.append({"role": "assistant", "content": welcome, "metrics": None})

    # Afficher les messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"],
                              avatar="🧭" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

            # Sources et métriques
            if msg["role"] == "assistant" and msg.get("metrics") and st.session_state.show_metrics:
                metrics = msg["metrics"]
                with st.expander("📊 Métriques RAGAS", expanded=False):
                    # Dans pages_chat.py, remplace les 4 colonnes par 5 :
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Fidélité", f"{metrics['faithfulness']:.0%}")
                    col2.metric("Pertinence", f"{metrics['answer_relevancy']:.0%}")
                    col3.metric("Précision ctx", f"{metrics['context_precision']:.0%}")
                    col4.metric("Rappel ctx", f"{metrics['context_recall']:.0%}")
                    col5.metric("Correctness", f"{metrics['answer_correctness']:.0%}")

    # ── Questions suggérées ───────────────────────────────────────────────────
    if len(st.session_state.messages) <= 1:
        st.markdown("**💡 Questions suggérées :**")
        cols = st.columns(2)
        for i, q in enumerate(QUESTIONS_SUGGEREES[:4]):
            with cols[i % 2]:
                if st.button(q, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()

    # ── Évaluation rapide ─────────────────────────────────────────────────────
    if len(st.session_state.messages) >= 3:
        last_assistant = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "assistant" and m.get("question")),
            None
        )
        if last_assistant and not last_assistant.get("evaluated"):
            with st.expander("⭐ Évaluer cette réponse", expanded=False):
                sat = st.slider("Satisfaction (1-5)", 1, 5, 4, key="sat_slider")
                col1, col2 = st.columns(2)
                with col1:
                    pertinent = st.checkbox("✅ Réponse pertinente", value=True)
                with col2:
                    personnalise = st.checkbox("🎯 Réponse personnalisée", value=True)
                commentaire = st.text_input("Commentaire (optionnel)", key="eval_comment")

                if st.button("Envoyer l'évaluation", type="primary"):
                    db.save_evaluation(
                        session_id=session_id,
                        question=last_assistant.get("question", ""),
                        reponse=last_assistant["content"],
                        satisfaction=sat,
                        pertinent=pertinent,
                        personnalise=personnalise,
                        commentaire=commentaire
                    )
                    last_assistant["evaluated"] = True
                    st.success("✅ Merci pour votre retour !")

    # ── Input utilisateur ─────────────────────────────────────────────────────
    # Vérifier si une question en attente (depuis bouton suggéré)
    pending = st.session_state.pop("pending_question", None)

    question = st.chat_input("Posez votre question sur l'orientation...") or pending

    if question:
        # Afficher la question utilisateur
        st.session_state.messages.append({"role": "user", "content": question})
        db.save_message(session_id, "user", question)

        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        # Générer la réponse
        with st.chat_message("assistant", avatar="🧭"):
            with st.spinner("OrientBot réfléchit à votre profil..."):
                if chain is None:
                    answer = (
                        "⚠️ Le système RAG n'est pas encore initialisé. "
                        "Veuillez indexer la base de connaissances d'abord.\n\n"
                        "**Mode démonstration** — voici ce que je ferais :\n"
                        f"Basé sur votre profil ({profil.prenom}, {profil.niveau}, "
                        f"intérêts : {', '.join(profil.interets)}), je rechercherais "
                        "les établissements les plus adaptés dans notre base de données."
                    )
                    source_docs = []
                else:
                    from rag.llm import generate_response
                    history = [m for m in st.session_state.messages if m["role"] in ["user","assistant"]]
                    #AJOUT
                    chain["profil_context"] = profil.to_context_string()
                    result = generate_response(chain, question, history)
                    answer = result["answer"]
                    source_docs = result["source_documents"]

                st.markdown(answer)

                # Calcul métriques RAGAS
                metrics = evaluate_response(question, answer, source_docs)
                db.save_ragas_metric(
                    session_id=session_id,
                    question=question,
                    faithfulness=metrics["faithfulness"],
                    answer_relevancy=metrics["answer_relevancy"],
                    context_precision=metrics["context_precision"],
                    context_recall=metrics["context_recall"],
                    answer_correctness=metrics["answer_correctness"]
                )

                # Sources
                if source_docs:
                    with st.expander(f"📚 {len(source_docs)} sources utilisées", expanded=False):
                        for i, doc in enumerate(source_docs[:3], 1):
                            src = doc.metadata.get("source", "inconnu")
                            st.markdown(f"**Source {i}** ({src})")
                            st.caption(doc.page_content[:300] + "...")

        # Sauvegarder
        db.save_message(session_id, "assistant", answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "metrics": metrics,
            "question": question,
            "evaluated": False
        })

        st.rerun()
