"""
Page Tableau de Bord — métriques et statistiques
"""
import streamlit as st
import plotly.graph_objects as go
from backend.database import DatabaseManager


def render_gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 13, "family": "sans-serif"}},
        number={"suffix": "%", "font": {"size": 22}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#d1fae5"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": value * 100
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_dashboard(db: DatabaseManager):
    st.markdown("## Tableau de Bord")
    st.markdown("Analyse des performances du système OrientBot.")

    stats       = db.get_stats()
    ragas_stats = db.get_ragas_stats()

    # ── KPIs ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    kpi_style = """
        background: white; border: 1px solid #e5e7eb;
        border-radius: 14px; padding: 1.2rem;
        text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    """
    with col1:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:2rem;">🧑‍🎓</div>
            <div style="font-size:1.8rem; font-weight:800; color:#059669;">
                {stats['total_sessions']}
            </div>
            <div style="font-size:0.8rem; color:#6b7280;">Sessions créées</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:2rem;">💬</div>
            <div style="font-size:1.8rem; font-weight:800; color:#1d4ed8;">
                {stats['total_messages']}
            </div>
            <div style="font-size:0.8rem; color:#6b7280;">Messages échangés</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        sat_color = "#059669" if stats['avg_satisfaction'] >= 4 else "#d97706"
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:2rem;">⭐</div>
            <div style="font-size:1.8rem; font-weight:800; color:{sat_color};">
                {stats['avg_satisfaction']:.1f}/5
            </div>
            <div style="font-size:0.8rem; color:#6b7280;">Satisfaction moyenne</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="{kpi_style}">
            <div style="font-size:2rem;">📊</div>
            <div style="font-size:1.8rem; font-weight:800; color:#7c3aed;">
                {stats['total_evaluations']}
            </div>
            <div style="font-size:0.8rem; color:#6b7280;">Évaluations reçues</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Métriques RAGAS ───────────────────────────────────────────────────
    st.markdown("### 🔬 Métriques RAGAS (Qualité du RAG)")
    st.caption("Basées sur BERTScore sémantique multilingue")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.plotly_chart(
            render_gauge(ragas_stats["faithfulness"],
                         "Fidélité", "#059669"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            render_gauge(ragas_stats["answer_relevancy"],
                         "Pertinence", "#1d4ed8"),
            use_container_width=True
        )
    with col3:
        st.plotly_chart(
            render_gauge(ragas_stats["context_precision"],
                         "Précision ctx", "#7c3aed"),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            render_gauge(ragas_stats.get("context_recall", 0),
                         "Rappel ctx", "#d97706"),
            use_container_width=True
        )
    with col5:
        st.plotly_chart(
            render_gauge(ragas_stats.get("answer_correctness", 0),
                         "Correctness", "#dc2626"),
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar RAGAS ───────────────────────────────────────────────────────
    st.markdown("### 📈 Évaluation globale")
    col1, col2 = st.columns(2)

    with col1:
        labels = [
            "Fidélité", "Pertinence",
            "Précision ctx", "Rappel ctx", "Correctness"
        ]
        values = [
            ragas_stats["faithfulness"],
            ragas_stats["answer_relevancy"],
            ragas_stats["context_precision"],
            ragas_stats.get("context_recall", 0),
            ragas_stats.get("answer_correctness", 0),
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(5,150,105,0.2)",
            line_color="#059669",
            line_width=2,
        ))
        fig_radar.update_layout(
            title="Radar des métriques RAGAS",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20, l=30, r=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        fig_bar = go.Figure(go.Bar(
            x=["Pertinence", "Personnalisation"],
            y=[stats["pct_pertinent"], stats["pct_personnalise"]],
            marker_color=["#059669", "#1d4ed8"],
            text=[f"{v:.0f}%" for v in [
                stats["pct_pertinent"], stats["pct_personnalise"]
            ]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Taux de satisfaction qualitative (%)",
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 110]),
            margin=dict(t=40, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── État du système ───────────────────────────────────────────────────
    st.markdown("### ⚙️ État du système")
    col1, col2 = st.columns(2)
    with col1:
        from app.config import CHROMA_DIR, MISTRAL_MODEL
        chroma_ok = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())
        st.markdown(f"""
        | Composant | Statut |
        |-----------|--------|
        | ChromaDB | {"✅ Indexé" if chroma_ok else "❌ Non indexé"} |
        | LLM | {MISTRAL_MODEL} |
        | Embedding | `paraphrase-multilingual-mpnet-base-v2` |
        | Évaluation | ✅ BERTScore + RAGAS |
        | Base SQLite | ✅ Active |
        """)
    with col2:
        from app.config import KNOWLEDGE_BASE_DIR
        files = list(KNOWLEDGE_BASE_DIR.glob("*")) \
            if KNOWLEDGE_BASE_DIR.exists() else []
        csv_c = sum(1 for f in files if f.suffix == ".csv")
        pdf_c = sum(1 for f in files if f.suffix == ".pdf")
        st.markdown(f"""
        | Ressource | Quantité |
        |-----------|----------|
        | Fichiers CSV | {csv_c} |
        | Fichiers PDF | {pdf_c} |
        | Total documents | {csv_c + pdf_c} |
        | Vecteurs ChromaDB | ~794 |
        | Top-K retrieval | 12 documents |
        """)