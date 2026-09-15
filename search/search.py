"""
Recherche hybride + reclassement : compare un profil (fichier markdown) aux
offres stockées dans PostgreSQL/pgvector, en combinant similarité
vectorielle et recherche par mots-clés (fusion RRF, voir hybrid_search.py),
puis reclasse le pool obtenu avec un cross-encoder local (voir rerank.py),
avec filtres optionnels sur type de contrat, localisation et expérience
requise.

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
from sentence_transformers import SentenceTransformer

from hybrid_search import EMBEDDING_MODEL_NAME, PG_CONFIG
from rerank import search_and_rerank


def parse_args():
    parser = argparse.ArgumentParser(description="Recherche hybride d'offres d'emploi")
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


def main():
    args = parse_args()

    print(f"Lecture du profil : {args.profil_path}")
    texte_profil = load_profil(args.profil_path)
    print(f"{len(texte_profil)} caractères chargés.\n")

    print(f"Chargement du modèle {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Modèle chargé.\n")

    print("Calcul de l'embedding du profil...")
    embedding_profil = model.encode(texte_profil)
    print("Embedding calculé.\n")

    if not PG_CONFIG["password"]:
        print("ERREUR : PG_PASSWORD manquant dans .env")
        sys.exit(1)

    print("Recherche hybride (vectorielle + mots-clés) + reclassement en cours "
          "(le modèle de reranking peut être téléchargé au premier lancement)...")
    resultats = search_and_rerank(
        texte_profil,
        embedding_profil,
        type_contrat=args.type_contrat,
        code_postal=args.code_postal,
        experience=args.experience,
        limit=args.limit,
    )

    if not resultats:
        print("Aucune offre ne correspond aux critères.")
        return

    print(f"\nTop {len(resultats)} offres les plus proches du profil :\n")
    for r in resultats:
        print(f"[rerank {r['score_rerank']:.4f} | RRF {r['score_rrf']:.4f} | distance {r['distance']:.4f} | "
              f"rang vecteur {r['rang_vectoriel'] or '-'} | rang mots-clés {r['rang_motscles'] or '-'}] "
              f"{r['intitule']}")
        print(f"    {r['entreprise_nom'] or 'N/A'} | {r['lieu_libelle'] or 'N/A'} | "
              f"{r['type_contrat_libelle'] or 'N/A'} | {r['experience_libelle'] or 'N/A'}")
        print(f"    id: {r['id']}\n")


if __name__ == "__main__":
    main()
