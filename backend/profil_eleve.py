"""
Modèle de données pour le profil de l'élève/étudiant
"""
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class NiveauEtudes(str, Enum):
    TRONC_COMMUN    = "Tronc Commun"
    PREMIERE_BAC    = "1ère Bac (Secondaire)"
    DEUXIEME_BAC    = "2ème Bac (Terminale)"
    BACHELIER       = "Bachelier (Bac obtenu)"
    ETUDIANT        = "Étudiant (en cours d'études)"


class TypeEtablissement(str, Enum):
    PUBLIC          = "Public"
    PRIVE           = "Privé"
    LES_DEUX        = "Les deux (pas de préférence)"


class SituationSocioEco(str, Enum):
    MODESTE         = "Modeste (budget limité)"
    MOYEN           = "Moyen"
    AISE            = "Aisé"


class ProfilEleve(BaseModel):
    # ── Infos générales ─────────────────────────────────────────────────────
    prenom:                 str             = Field(..., description="Prénom de l'élève")
    niveau:                 NiveauEtudes    = Field(..., description="Niveau scolaire actuel")

    # ── Bac (si bachelier) ──────────────────────────────────────────────────
    type_bac:               Optional[str]   = Field(None, description="Type de baccalauréat")
    moyenne_bac:            Optional[float] = Field(None, ge=0, le=20, description="Moyenne générale du bac /20")
    mention_bac:            Optional[str]   = Field(None, description="Mention obtenue")

    # ── Notes par matière (si bachelier ou secondaire) ──────────────────────
    note_maths:             Optional[float] = Field(None, ge=0, le=20)
    note_physique:          Optional[float] = Field(None, ge=0, le=20)
    note_svt:               Optional[float] = Field(None, ge=0, le=20)
    note_francais:          Optional[float] = Field(None, ge=0, le=20)
    note_anglais:           Optional[float] = Field(None, ge=0, le=20)
    note_philosophie:       Optional[float] = Field(None, ge=0, le=20)
    note_arabe:             Optional[float] = Field(None, ge=0, le=20)
    note_economie:          Optional[float] = Field(None, ge=0, le=20)

    # ── Intérêts & Objectifs ─────────────────────────────────────────────────
    interets:               List[str]       = Field(default_factory=list, description="Domaines d'intérêt")
    objectif_carriere:      Optional[str]   = Field(None, description="Métier ou domaine cible")
    description_objectif:   Optional[str]   = Field(None, description="Description libre des objectifs")

    # ── Préférences ─────────────────────────────────────────────────────────
    ville_preferee:         Optional[str]   = Field(None, description="Ville souhaitée pour étudier")
    type_etablissement:     TypeEtablissement = Field(TypeEtablissement.LES_DEUX)
    situation_socio_eco:    SituationSocioEco = Field(SituationSocioEco.MOYEN)
    budget_annuel_max:      Optional[int]   = Field(None, description="Budget max en MAD/an")

    # ── Infos complémentaires ─────────────────────────────────────────────────
    mobilite:               bool            = Field(True, description="Prêt à quitter sa ville ?")
    langue_preferee:        str             = Field("Français", description="Langue de réponse préférée")

    class Config:
        use_enum_values = True

    def to_context_string(self) -> str:
        """Génère un résumé du profil pour le prompt LLM."""
        lignes = [f"PROFIL DE L'ÉLÈVE : {self.prenom}"]
        lignes.append(f"- Niveau : {self.niveau}")

        if self.type_bac:
            lignes.append(f"- Type de Bac : {self.type_bac}")
        if self.moyenne_bac:
            mention = self.mention_bac or ""
            lignes.append(f"- Moyenne Bac : {self.moyenne_bac}/20 ({mention})")

        # Notes
        notes = []
        mapping = {
            "Mathématiques": self.note_maths,
            "Physique-Chimie": self.note_physique,
            "SVT": self.note_svt,
            "Français": self.note_francais,
            "Anglais": self.note_anglais,
            "Philosophie": self.note_philosophie,
            "Arabe": self.note_arabe,
            "Économie": self.note_economie,
        }
        for matiere, note in mapping.items():
            if note is not None:
                notes.append(f"{matiere}: {note}/20")
        if notes:
            lignes.append(f"- Notes : {', '.join(notes)}")

        if self.interets:
            lignes.append(f"- Centres d'intérêt : {', '.join(self.interets)}")
        if self.objectif_carriere:
            lignes.append(f"- Objectif de carrière : {self.objectif_carriere}")
        if self.description_objectif:
            lignes.append(f"- Description : {self.description_objectif}")

        lignes.append(f"- Ville souhaitée : {self.ville_preferee or 'Flexible'}")
        lignes.append(f"- Type d'établissement : {self.type_etablissement}")
        lignes.append(f"- Situation socio-économique : {self.situation_socio_eco}")
        if self.budget_annuel_max:
            lignes.append(f"- Budget max : {self.budget_annuel_max:,} MAD/an")
        lignes.append(f"- Mobilité géographique : {'Oui' if self.mobilite else 'Non'}")

        return "\n".join(lignes)
