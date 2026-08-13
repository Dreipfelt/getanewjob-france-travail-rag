"""
Recherche sémantique : compare un profil (fichier markdown) aux offres
stockées dans PostgreSQL/pgvector, avec filtres optionnels sur type de
contrat, localisation et expérience requise.

Usage :
    python search.py chemin/vers/profil.md
    python search.py chemin/vers/profil.md --type-contrat CDI --code-postal 75

Prérequis :
    pip install sentence-transformers psycopg2-binary python-dotenv pgvector

Fichier .env attendu (racine du projet, résolu automatiquement) :
    PG_HOST=localhost
    PG_PORT=5432
    PG_DB=getanewjob
    PG_USER=getanewjob
    PG_PASSWORD=...
    (optionnel) HF_TOKEN=...  -> accélère le téléchargement du modèle,
    n'affecte pas la qualité des résultats de recherche.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
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

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Recherche sémantique d'offres d'emploi")
    parser.add_argument("profil_path", help="Chemin vers le fichier markdown du profil/CV")
    parser.add_argument("--type-contrat", help="Filtrer par type de contrat (ex: CDI)")
    parser.add_argument("--code-postal", help="Filtrer par département (2 premiers chiffres du code postal)")
    parser.add_argument("--experience", help="Filtrer par code d'expérience exigée")
    parser.add_argument("--limit", type=int, default=10, help="Nombre de résultats à afficher (défaut: 10)")
    return parser.parse_args()


def load_profil(path: str) -> str:
    if not os.path.exists(path):
        print(f"ERREUR : fichier introuvable : {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_query(args) -> tuple:
    """
    Construit dynamiquement la clause WHERE et ses paramètres selon les
    filtres fournis en ligne de commande. Retourne (clause_sql, params).
    """
    conditions = []
    params = []

    if args.type_contrat:
        conditions.append("type_contrat = %s")
        params.append(args.type_contrat)

    if args.code_postal:
        conditions.append("lieu_code_postal LIKE %s")
        params.append(f"{args.code_postal}%")

    if args.experience:
        conditions.append("experience_exige = %s")
        params.append(args.experience)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where_clause, params


def main():
    args = parse_args()

    print(f"Lecture du profil : {args.profil_path}")
    texte_profil = load_profil(args.profil_path)
    print(f"{len(texte_profil)} caractères chargés.\n")

    print(f"Chargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Modèle chargé.\n")

    print("Calcul de l'embedding du profil...")
    embedding_profil = model.encode(texte_profil)
    print("Embedding calculé.\n")

    if not PG_CONFIG["password"]:
        print("ERREUR : PG_PASSWORD manquant dans .env")
        sys.exit(1)

    conn = psycopg2.connect(**PG_CONFIG)
    register_vector(conn)  # active la conversion automatique vector <-> numpy/list
    cur = conn.cursor()

    where_clause, params = build_query(args)

    # Distance cosinus (<=>), cohérente avec l'index HNSW créé dans schema.sql
    # (vector_cosine_ops). Plus la distance est proche de 0, plus l'offre
    # est sémantiquement proche du profil.
    query = f"""
        SELECT id, intitule, entreprise_nom, lieu_libelle,
               type_contrat_libelle, experience_libelle,
               embedding <=> %s AS distance
        FROM offres
        {where_clause}
        ORDER BY distance ASC
        LIMIT %s
    """

    cur.execute(query, [embedding_profil] + params + [args.limit])
    resultats = cur.fetchall()

    cur.close()
    conn.close()

    if not resultats:
        print("Aucune offre ne correspond aux critères.")
        return

    print(f"Top {len(resultats)} offres les plus proches du profil :\n")
    for row in resultats:
        offre_id, intitule, entreprise, lieu, contrat, experience, distance = row
        print(f"[{distance:.4f}] {intitule}")
        print(f"    {entreprise or 'N/A'} | {lieu or 'N/A'} | {contrat or 'N/A'} | {experience or 'N/A'}")
        print(f"    id: {offre_id}\n")


if __name__ == "__main__":
    main()
