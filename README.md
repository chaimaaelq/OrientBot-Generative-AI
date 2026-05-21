# OrientBot — Chatbot Conseiller d'Orientation Scolaire
## Projet de Fin d'Études — Master Big Data & IA

### Démarrage rapide

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Ajouter MISTRAL_API_KEY
python -m knowledge_base.indexer
streamlit run app/main.py
```

### Structure
- app/ : Streamlit (main.py, profil, chat, dashboard)
- backend/ : profil_eleve.py (Pydantic), database.py (SQLite)
- rag/ : vector_store.py (ChromaDB), llm.py (Mistral RAG)
- knowledge_base/ : loader.py, indexer.py, documents/
- evaluation/ : ragas_eval.py (Faithfulness, Relevancy, Precision)

### Obtenir la clé Mistral AI (gratuite)
https://console.mistral.ai → API Keys
