"""
API FastAPI exposant :
- POST /search : recherche sémantique (avec filtres optionnels), gratuite
- POST /score  : scoring LLM motivé sur un ensemble d'offres (payant, appelé
  à la demande uniquement)

Lancement :
    uvicorn api:app --reload --port 8000

Prérequis :
    pip install fastapi uvicorn sentence-transformers psycopg2-binary
    pip install python-dotenv pgvector mistralai

Réutilise la même config (.env) que les scripts CLI existants.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from mistralai.client import Mistral

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# search/ n'est pas packagé (pas de setup.py) : on ajoute son dossier au
# path pour réutiliser la même logique de recherche hybride que les CLI,
# plutôt que de la dupliquer une troisième fois.
sys.path.insert(0, str(PROJECT_ROOT / "search"))
from hybrid_search import EMBEDDING_MODEL_NAME, PG_CONFIG, get_connection, hybrid_search  # noqa: E402

MISTRAL_MODEL = "mistral-small-latest"
CACHE_PATH = PROJECT_ROOT / "cache_scoring.json"
MAX_DESCRIPTION_LEN = 1500

app = FastAPI(title="GetANewJob API")

# Le modèle d'embedding est coûteux à charger : une seule fois au démarrage
# du serveur, pas à chaque requête.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# --- Schémas de requête/réponse ---

class SearchRequest(BaseModel):
    profil: str
    types_contrat: Optional[list[str]] = None
    departements: Optional[list[str]] = None
    experience: Optional[str] = None
    limit: int = 10


class OffreResult(BaseModel):
    id: str
    intitule: str
    entreprise_nom: Optional[str]
    lieu_libelle: Optional[str]
    type_contrat_libelle: Optional[str]
    experience_libelle: Optional[str]
    distance: float
    score_rrf: float
    rang_vectoriel: Optional[int]
    rang_motscles: Optional[int]


class SearchResponse(BaseModel):
    resultats: list[OffreResult]


class ScoreRequest(BaseModel):
    profil: str
    types_contrat: Optional[list[str]] = None
    departements: Optional[list[str]] = None
    experience: Optional[str] = None
    top_n: int = 10


class ScoreResult(BaseModel):
    id: str
    intitule: str
    entreprise_nom: Optional[str]
    lieu_libelle: Optional[str]
    score: int
    points_forts: list[str]
    points_faibles: list[str]
    red_flags: list[str]


class ScoreResponse(BaseModel):
    resultats: list[ScoreResult]
    appels_api: int
    servis_depuis_cache: int


# --- Fonctions internes ---

def get_db_connection():
    try:
        return get_connection()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def score_offre_llm(client: Mistral, profil_texte: str, offre: dict) -> Optional[dict]:
    description_tronquee = (offre.get("description") or "")[:MAX_DESCRIPTION_LEN]

    prompt = f"""Tu es un assistant de tri de candidatures. Compare le profil ci-dessous à l'offre d'emploi, et évalue leur adéquation.

PROFIL DU CANDIDAT :
{profil_texte}

OFFRE D'EMPLOI :
Intitulé : {offre['intitule']}
Entreprise : {offre.get('entreprise_nom') or 'Non précisé'}
Lieu : {offre.get('lieu_libelle') or 'Non précisé'}
Contrat : {offre.get('type_contrat_libelle') or 'Non précisé'}
Expérience requise : {offre.get('experience_libelle') or 'Non précisé'}
Description : {description_tronquee}

Réponds UNIQUEMENT avec un objet JSON (rien d'autre, pas de texte avant/après), au format exact suivant :
{{
  "score": <entier de 0 à 100>,
  "points_forts": ["<point 1>", "<point 2>", ...],
  "points_faibles": ["<point 1>", "<point 2>", ...],
  "red_flags": ["<éventuel signal d'alerte>", ...]
}}

Le score doit refléter l'adéquation réelle entre les compétences/expérience du candidat et les exigences de l'offre. Sois honnête et nuancé, pas complaisant."""

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )

    texte_reponse = response.choices[0].message.content.strip()
    if texte_reponse.startswith("```"):
        texte_reponse = texte_reponse.strip("`")
        if texte_reponse.startswith("json"):
            texte_reponse = texte_reponse[4:]
        texte_reponse = texte_reponse.strip()

    try:
        return json.loads(texte_reponse)
    except json.JSONDecodeError:
        return None


# --- Endpoints ---

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Recherche hybride (vectorielle + mots-clés) gratuite, sans appel LLM."""
    if not PG_CONFIG["password"]:
        raise HTTPException(status_code=500, detail="PG_PASSWORD manquant dans .env")
    embedding_profil = embedding_model.encode(req.profil)
    rows = hybrid_search(
        req.profil, embedding_profil,
        types_contrat=req.types_contrat, departements=req.departements,
        experience=req.experience, limit=req.limit,
    )
    return {"resultats": rows}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Scoring LLM motivé sur le top-N. Appelle l'API Mistral (coût réel)."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY manquant dans .env")
    if not PG_CONFIG["password"]:
        raise HTTPException(status_code=500, detail="PG_PASSWORD manquant dans .env")

    embedding_profil = embedding_model.encode(req.profil)
    offres = hybrid_search(
        req.profil, embedding_profil,
        types_contrat=req.types_contrat, departements=req.departements,
        experience=req.experience, limit=req.top_n, with_description=True,
    )

    if not offres:
        return {"resultats": [], "appels_api": 0, "servis_depuis_cache": 0}

    profil_hash = hashlib.sha256(req.profil.encode()).hexdigest()[:16]
    client = Mistral(api_key=api_key)
    cache = load_cache()

    resultats = []
    appels_api = 0
    servis_cache = 0

    for offre in offres:
        key = f"{profil_hash}:{offre['id']}"
        if key in cache:
            score_data = cache[key]
            servis_cache += 1
        else:
            score_data = score_offre_llm(client, req.profil, offre)
            appels_api += 1
            if score_data:
                cache[key] = score_data

        if score_data:
            resultats.append({**offre, **score_data})

    save_cache(cache)
    resultats.sort(key=lambda r: r["score"], reverse=True)

    return {
        "resultats": resultats,
        "appels_api": appels_api,
        "servis_depuis_cache": servis_cache,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/departements")
def departements():
    """
    Retourne les départements distincts (2 premiers chiffres du code postal)
    réellement présents en base, triés. Permet à l'interface de ne jamais
    afficher un filtre désynchronisé des vraies données.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT LEFT(lieu_code_postal, 2) AS departement
        FROM offres
        WHERE lieu_code_postal IS NOT NULL
        ORDER BY departement
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"departements": [r[0] for r in rows]}
