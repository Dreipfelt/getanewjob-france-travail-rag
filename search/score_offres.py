"""
Scoring LLM : pour chaque offre du top-N (issu de la recherche sémantique),
demande à Mistral d'évaluer la pertinence par rapport au profil, avec un
score et une justification structurée.

Usage :
    python score_offres.py frederic_tellier_cv.md
    python score_offres.py frederic_tellier_cv.md --type-contrat CDI --top-n 10

Prérequis :
    pip install mistralai sentence-transformers psycopg2-binary python-dotenv pgvector

Fichier .env (ajouter à celui existant) :
    MISTRAL_API_KEY=votre_clé

Maîtrise des coûts :
- Le scoring ne s'applique qu'au top-N déjà filtré par recherche sémantique
  (pas sur les 795 offres).
- Un cache local (fichier JSON) évite de re-scorer une offre déjà scorée
  lors d'exécutions répétées pendant le développement/debug.
- Le texte envoyé au LLM est résumé (pas le CV complet, pas la description
  brute intégrale) pour limiter les tokens d'entrée.
"""

import os
import sys
import json
import time
import random
import argparse
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from mistralai.client import Mistral

from hybrid_search import EMBEDDING_MODEL_NAME, PG_CONFIG, hybrid_search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = EMBEDDING_MODEL_NAME
MISTRAL_MODEL = "mistral-small-latest"
CACHE_PATH = PROJECT_ROOT / "cache_scoring.json"

# Longueur max de la description d'offre envoyée au LLM (caractères).
# Réduit le coût sans perdre l'essentiel du contenu.
MAX_DESCRIPTION_LEN = 1500


def parse_args():
    parser = argparse.ArgumentParser(description="Scoring LLM des offres les plus proches du profil")
    parser.add_argument("profil_path", help="Chemin vers le fichier markdown du profil/CV")
    parser.add_argument("--type-contrat", help="Filtrer par type de contrat (ex: CDI)")
    parser.add_argument("--code-postal", help="Filtrer par département (2 premiers chiffres)")
    parser.add_argument("--experience", help="Filtrer par code d'expérience exigée (D ou E)")
    parser.add_argument("--top-n", type=int, default=10, help="Nombre d'offres à scorer (défaut: 10)")
    return parser.parse_args()


def load_profil(path: str) -> str:
    if not os.path.exists(path):
        print(f"ERREUR : fichier introuvable : {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def cache_key(profil_hash: str, offre_id: str) -> str:
    return f"{profil_hash}:{offre_id}"


def score_offre(client: Mistral, profil_texte: str, offre: dict,
                max_retries: int = 4) -> dict:
    """
    Appelle Mistral pour scorer une offre par rapport au profil.
    Retry avec backoff exponentiel + jitter en cas de rate limit (429)
    ou d'erreur serveur transitoire (5xx).
    """
    description_tronquee = (offre["description"] or "")[:MAX_DESCRIPTION_LEN]

    prompt = f"""Tu es un assistant de tri de candidatures. Compare le profil ci-dessous à l'offre d'emploi, et évalue leur adéquation.

PROFIL DU CANDIDAT :
{profil_texte}

OFFRE D'EMPLOI :
Intitulé : {offre['intitule']}
Entreprise : {offre['entreprise_nom'] or 'Non précisé'}
Lieu : {offre['lieu_libelle'] or 'Non précisé'}
Contrat : {offre['type_contrat_libelle'] or 'Non précisé'}
Expérience requise : {offre['experience_libelle'] or 'Non précisé'}
Description : {description_tronquee}

Réponds UNIQUEMENT avec un objet JSON (rien d'autre, pas de texte avant/après), au format exact suivant :
{{
  "score": <entier de 0 à 100>,
  "points_forts": ["<point 1>", "<point 2>", ...],
  "points_faibles": ["<point 1>", "<point 2>", ...],
  "red_flags": ["<éventuel signal d'alerte>", ...]
}}

Le score doit refléter l'adéquation réelle entre les compétences/expérience du candidat et les exigences de l'offre. Sois honnête et nuancé, pas complaisant."""

    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )

            texte_reponse = response.choices[0].message.content.strip()

            # Mistral enveloppe parfois sa réponse dans des balises markdown
            if texte_reponse.startswith("```"):
                texte_reponse = texte_reponse.strip("`")
                if texte_reponse.startswith("json"):
                    texte_reponse = texte_reponse[4:]
                texte_reponse = texte_reponse.strip()

            try:
                return json.loads(texte_reponse)
            except json.JSONDecodeError:
                print(f"  AVERTISSEMENT : réponse non-JSON pour {offre['id']}")
                print(f"  Réponse brute : {texte_reponse[:200]}")
                return None

        except Exception as e:
            # Détection du rate limit sur le message d'erreur (robuste
            # quelle que soit la version de mistralai)
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            is_server_error = any(c in str(e) for c in ["500", "502", "503", "504"])

            if (is_rate_limit or is_server_error) and attempt < max_retries - 1:
                # Backoff exponentiel : 2^attempt secondes + jitter aléatoire
                # 0.5s → 1s → 2s → 4s (+ 0-1s aléatoire à chaque fois)
                wait = (2 ** attempt) * 0.5 + random.uniform(0, 1)
                print(f"  Rate limit / erreur serveur sur {offre['id']} "
                      f"(tentative {attempt + 1}/{max_retries}). "
                      f"Attente {wait:.1f}s...")
                time.sleep(wait)
            else:
                # Erreur non retriable (auth, 404, etc.) ou max retries atteint
                print(f"  ERREUR non retriable sur {offre['id']} : {e}")
                return None

    return None  # max_retries épuisés sans succès


def main():
    args = parse_args()

    print(f"Lecture du profil : {args.profil_path}")
    texte_profil = load_profil(args.profil_path)
    profil_hash = hashlib.sha256(texte_profil.encode()).hexdigest()[:16]
    print(f"{len(texte_profil)} caractères chargés.\n")

    print(f"Chargement du modèle d'embedding...")
    model = SentenceTransformer(MODEL_NAME)
    embedding_profil = model.encode(texte_profil)
    print("Embedding calculé.\n")

    if not PG_CONFIG["password"]:
        print("ERREUR : PG_PASSWORD manquant dans .env")
        sys.exit(1)

    print(f"Récupération du top-{args.top_n} par recherche hybride (vectorielle + mots-clés)...")
    offres = hybrid_search(
        texte_profil,
        embedding_profil,
        type_contrat=args.type_contrat,
        code_postal=args.code_postal,
        experience=args.experience,
        limit=args.top_n,
        with_description=True,
    )
    print(f"{len(offres)} offres à scorer.\n")

    if not offres:
        print("Aucune offre ne correspond aux critères.")
        return

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("ERREUR : MISTRAL_API_KEY manquant dans .env")
        sys.exit(1)

    client = Mistral(api_key=api_key)
    cache = load_cache()

    resultats = []
    appels_api = 0

    for offre in offres:
        key = cache_key(profil_hash, offre["id"])

        if key in cache:
            print(f"  [cache] {offre['intitule']}")
            score_data = cache[key]
        else:
            print(f"  [API]   {offre['intitule']}")
            score_data = score_offre(client, texte_profil, offre)
            appels_api += 1
            if score_data:
                cache[key] = score_data

        if score_data:
            resultats.append({**offre, **score_data})

    save_cache(cache)
    print(f"\n{appels_api} appel(s) API effectué(s), {len(offres) - appels_api} servi(s) depuis le cache.\n")

    # Tri par score de scoring LLM (pas par distance sémantique)
    resultats.sort(key=lambda r: r["score"], reverse=True)

    print("=" * 70)
    print("RÉSULTATS TRIÉS PAR SCORE DE PERTINENCE")
    print("=" * 70)
    for r in resultats:
        print(f"\n[{r['score']}/100] {r['intitule']}")
        print(f"  {r['entreprise_nom'] or 'N/A'} | {r['lieu_libelle'] or 'N/A'}")
        print(f"  Points forts   : {', '.join(r['points_forts']) if r['points_forts'] else '-'}")
        print(f"  Points faibles : {', '.join(r['points_faibles']) if r['points_faibles'] else '-'}")
        if r.get("red_flags"):
            print(f"  ⚠ Red flags    : {', '.join(r['red_flags'])}")


if __name__ == "__main__":
    main()
