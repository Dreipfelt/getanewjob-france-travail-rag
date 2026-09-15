"""
Recherche hybride : combine recherche vectorielle (pgvector, similarité
cosinus) et recherche par mots-clés (PostgreSQL full-text search), fusionnées
par Reciprocal Rank Fusion (RRF).

Pourquoi hybride :
- La recherche vectorielle seule capture bien la proximité sémantique globale
  entre un profil et une offre, mais peut diluer une correspondance exacte de
  terme précis (nom de techno, intitulé de poste) noyée dans un profil long.
- La recherche par mots-clés seule capture ces correspondances exactes mais
  ignore synonymes/paraphrases.
- RRF combine les deux classements par le RANG de chaque offre dans chaque
  branche (pas par la valeur brute du score), ce qui évite d'avoir à
  normaliser des échelles incomparables (distance cosinus vs ts_rank).
  Constante k=60 : valeur standard de la littérature (Cormack et al., 2009).

Le profil (souvent un CV entier) est réduit, pour la branche mots-clés, à
l'ensemble de ses lexèmes normalisés (racinisation française) combinés en
OR : on cherche les offres qui partagent le plus de termes avec le profil,
plutôt que d'exiger la présence de tous les mots (ce que ferait
plainto_tsquery sur un texte aussi long).

Module partagé par search.py (CLI), score_offres.py (CLI) et api.py
(FastAPI) pour ne pas dupliquer la logique de filtrage/fusion à trois
endroits.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DB", "getanewjob"),
    "user": os.getenv("PG_USER", "getanewjob"),
    "password": os.getenv("PG_PASSWORD"),
}

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

RRF_K = 60


def get_connection():
    if not PG_CONFIG["password"]:
        raise RuntimeError("PG_PASSWORD manquant dans .env")
    conn = psycopg2.connect(**PG_CONFIG)
    register_vector(conn)
    return conn


def build_conditions(types_contrat=None, departements=None,
                      type_contrat=None, code_postal=None, experience=None):
    """
    Construit les conditions WHERE communes aux deux branches de la
    recherche hybride. Accepte soit des listes (API/Streamlit :
    types_contrat, departements), soit des valeurs uniques (CLI :
    type_contrat, code_postal), pour rester compatible avec les deux usages
    existants. Retourne (conditions: list[str], params: list).
    """
    conditions = []
    params = []

    if types_contrat:
        conditions.append("type_contrat = ANY(%s)")
        params.append(types_contrat)
    elif type_contrat:
        conditions.append("type_contrat = %s")
        params.append(type_contrat)

    if departements:
        or_conditions = " OR ".join(["lieu_code_postal LIKE %s"] * len(departements))
        conditions.append(f"({or_conditions})")
        params.extend([f"{dep}%" for dep in departements])
    elif code_postal:
        conditions.append("lieu_code_postal LIKE %s")
        params.append(f"{code_postal}%")

    if experience:
        conditions.append("experience_exige = %s")
        params.append(experience)

    return conditions, params


def _vector_search(cur, embedding, conditions, params, k):
    """Retourne [(id, distance), ...] trié par distance cosinus croissante."""
    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    query = f"""
        SELECT id, embedding <=> %s AS distance
        FROM offres
        {where_clause}
        ORDER BY distance ASC
        LIMIT %s
    """
    cur.execute(query, [embedding] + params + [k])
    return cur.fetchall()


def _keyword_search(cur, profil_texte, conditions, params, k):
    """Retourne [(id, ts_rank), ...] trié par pertinence mots-clés décroissante."""
    kw_conditions = conditions + [
        "l.tsq IS NOT NULL",
        "o.texte_embedding_tsv @@ to_tsquery('french', l.tsq)",
    ]
    where_clause = "WHERE " + " AND ".join(kw_conditions)
    query = f"""
        WITH lexemes AS (
            SELECT string_agg(DISTINCT lexeme, ' | ') AS tsq
            FROM unnest(to_tsvector('french', %s))
        )
        SELECT o.id, ts_rank(o.texte_embedding_tsv, to_tsquery('french', l.tsq)) AS rang
        FROM offres o, lexemes l
        {where_clause}
        ORDER BY rang DESC
        LIMIT %s
    """
    cur.execute(query, [profil_texte] + params + [k])
    return cur.fetchall()


def reciprocal_rank_fusion(vector_results, keyword_results, k=RRF_K):
    """
    vector_results / keyword_results : listes ordonnées de tuples
    (id, score_brut), le premier élément étant le plus pertinent selon sa
    méthode. Retourne (fused_scores, vector_ranks, keyword_ranks) : trois
    dicts {id: valeur}, les deux derniers exposant le rang brut de chaque
    branche (utile pour l'affichage/debug), absents si l'offre n'y figure
    pas.
    """
    fused_scores = {}
    vector_ranks = {}
    keyword_ranks = {}

    for rank, (offre_id, _) in enumerate(vector_results, start=1):
        fused_scores[offre_id] = fused_scores.get(offre_id, 0.0) + 1.0 / (k + rank)
        vector_ranks[offre_id] = rank

    for rank, (offre_id, _) in enumerate(keyword_results, start=1):
        fused_scores[offre_id] = fused_scores.get(offre_id, 0.0) + 1.0 / (k + rank)
        keyword_ranks[offre_id] = rank

    return fused_scores, vector_ranks, keyword_ranks


def hybrid_search(profil_texte, embedding_profil, *, types_contrat=None,
                   departements=None, type_contrat=None, code_postal=None,
                   experience=None, limit=10, candidate_k=None,
                   with_description=False):
    """
    Recherche hybride : fusionne recherche vectorielle et recherche par
    mots-clés (RRF), puis renvoie les `limit` meilleures offres avec leurs
    scores de composantes pour transparence (distance cosinus, rang
    mots-clés, score RRF final).

    candidate_k : taille du top-K retenu dans CHAQUE branche avant fusion
    (sur-échantillonnage). Par défaut max(limit * 5, 50) pour laisser à la
    fusion de quoi arbitrer.

    Si la branche mots-clés échoue (ex : tsquery mal formée sur une entrée
    inhabituelle), la recherche se dégrade silencieusement en vectoriel pur
    plutôt que de faire échouer toute la requête.
    """
    candidate_k = candidate_k or max(limit * 5, 50)
    conditions, params = build_conditions(
        types_contrat, departements, type_contrat, code_postal, experience
    )

    conn = get_connection()
    try:
        cur = conn.cursor()

        vector_results = _vector_search(cur, embedding_profil, conditions, params, candidate_k)

        try:
            keyword_results = _keyword_search(cur, profil_texte, conditions, params, candidate_k)
        except psycopg2.Error:
            conn.rollback()
            keyword_results = []

        fused_scores, vector_ranks, keyword_ranks = reciprocal_rank_fusion(
            vector_results, keyword_results
        )
        top_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:limit]

        if not top_ids:
            return []

        colonnes = "id, intitule, entreprise_nom, lieu_libelle, type_contrat_libelle, experience_libelle"
        if with_description:
            colonnes += ", description"

        cur.execute(
            f"SELECT {colonnes}, embedding <=> %s AS distance FROM offres WHERE id = ANY(%s)",
            [embedding_profil, top_ids],
        )
        rows_by_id = {row[0]: row for row in cur.fetchall()}
        cur.close()
    finally:
        conn.close()

    noms_colonnes = colonnes.split(", ") + ["distance"]
    resultats = []
    for offre_id in top_ids:  # préserve l'ordre de fusion RRF
        row = rows_by_id.get(offre_id)
        if row is None:
            continue
        offre = dict(zip(noms_colonnes, row))
        offre["score_rrf"] = fused_scores[offre_id]
        offre["rang_vectoriel"] = vector_ranks.get(offre_id)
        offre["rang_motscles"] = keyword_ranks.get(offre_id)
        resultats.append(offre)
    return resultats
