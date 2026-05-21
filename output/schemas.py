"""
Schémas Pydantic pour les réponses du chatbot
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class EtablissementRecommandé(BaseModel):
    nom: str
    ville: str
    type: str  # Public / Privé
    filiere: str
    conditions: Optional[str] = None
    seuil: Optional[float] = None
    frais_annuels: Optional[str] = None
    justification: str


class ChatbotResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    recommandations: List[EtablissementRecommandé] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None
