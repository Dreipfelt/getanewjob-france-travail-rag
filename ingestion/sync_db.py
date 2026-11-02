"""
Synchronisation base <-> dernier export (format commun).

Lit data_raw/offres_latest.json, produit par ingestion/run_ingestion.py.

Pour chaque source présente dans l'export :
1. Supprime en base les offres de cette source qui ne sont plus dans l'export.
2. Recalcule les embeddings des offres nouvelles ou modifiées.
3. Upsert les offres.

Les sources absentes de l'export ne sont pas touchées.
Garde-fou : si l'export d'une source chute de plus de 50 % par rapport à la base,
le script s'arrête. Option --force pour passer outre.

Usage (depuis la racine du projet) :
    python -m ingestion.sync_db
    python -m ingestion.sync_db --force
"""

import os
import sys
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

INPUT_JSON = ROOT / "data_raw" / "offres_latest.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
SEUIL_CHUTE = 0.5
SOURCES_CONNUES = {"france_travail", "csp", "forem"}

PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DB", "getanewjob"),
    "user": os.getenv("PG_USER", "getanewjob"),
    "password": os.getenv("PG_PASSWORD"),
}

UPSERT_SQL = """
    INSERT INTO offres (
        id, source, intitule, description, entreprise_nom,
        lieu_libelle, lieu_code_postal, type_contrat, url,
        date_creation, raw_json, texte_embedding, embedding
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
    ON CONFLICT (id) DO UPDATE SET
        source = EXCLUDED.source,
        intitule = EXCLUDED.intitule,
        description = EXCLUDED.description,
        entreprise_nom = EXCLUDED.entreprise_nom,
        lieu_libelle = EXCLUDED.lieu_libelle,
        lieu_code_postal = EXCLUDED.lieu_code_postal,
        type_contrat = EXCLUDED.type_contrat,
        url = EXCLUDED.url,
        date_creation = EXCLUDED.date_creation,
        raw_json = EXCLUDED.raw_json,
        texte_embedding = EXCLUDED.texte_embedding,
        embedding = EXCLUDED.embedding,
        date_extraction = now()
"""

DATE_FORMATS = ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"]


def parse_date(s):
    """Accepte les formats des trois sources. Renvoie None si illisible."""
    if not s:
        return None
    s = str(s).strip()
    # Python < 3.11 n'accepte pas un fuseau sans deux-points (+0200)
    if len(s) >= 5 and s[-5] in "+-" and s[-3] != ":":
        s = s[:-2] + ":" + s[-2:]
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def build_texte_embedding(o: dict) -> str:
    parties = [o.get("intitule") or "", o.get("description") or ""]
    return "\n\n".join(p for p in parties if p.strip())


def row_db(o: dict, texte: str) -> tuple:
    return (
        o["id"],
        o["source"],
        o.get("intitule") or "",
        o.get("description") or None,
        o.get("entreprise"),
        o.get("lieu"),
        o.get("code_postal"),
        o.get("type_contrat"),
        o.get("url"),
        parse_date(o.get("date_publication")),
        json.dumps(o, ensure_ascii=False),
        texte,
    )


def main():
    force = "--force" in sys.argv

    if not INPUT_JSON.exists():
        sys.exit(f"ERREUR : fichier introuvable : {INPUT_JSON}\nLancez d'abord : python -m ingestion.run_ingestion")

    offres = json.loads(INPUT_JSON.read_text(encoding="utf-8"))["offres"]
    print(f"{len(offres)} offres dans le dernier export.")

    # --- Vérifications du format ---
    for o in offres:
        if "source" not in o or "id" not in o:
            sys.exit("ERREUR : format non reconnu (ancien export ?). Relancez run_ingestion.py.")
        if o["source"] not in SOURCES_CONNUES:
            sys.exit(f"ERREUR : source inconnue : {o['source']}")
        if str(o["id"]).startswith("ft:"):
            sys.exit("ERREUR : ids France Travail préfixés 'ft:'. Corrigez france_travail.py (id=o['id']) puis relancez run_ingestion.py.")

    if not PG_CONFIG["password"]:
        sys.exit("ERREUR : PG_PASSWORD manquant dans .env")

    par_source = defaultdict(dict)
    for o in offres:
        par_source[o["source"]][o["id"]] = o

    conn = psycopg2.connect(**PG_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    # --- Colonnes requises (migration 003) ---
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'offres'")
    colonnes = {r[0] for r in cur.fetchall()}
    manquantes = {"source", "url"} - colonnes
    if manquantes:
        sys.exit(f"ERREUR : colonnes manquantes en base : {manquantes}. Appliquez sql/003_ajout_source_url.sql.")

    # --- Garde-fou contre les exports partiels ---
    cur.execute("SELECT source, COUNT(*) FROM offres GROUP BY source")
    en_base = dict(cur.fetchall())
    for s, contenu in par_source.items():
        n_base, n_export = en_base.get(s, 0), len(contenu)
        if n_base and n_export < n_base * (1 - SEUIL_CHUTE) and not force:
            sys.exit(
                f"ARRÊT : {s} n'a que {n_export} offres dans l'export contre {n_base} en base "
                f"(chute > {int(SEUIL_CHUTE * 100)} %). Export probablement incomplet. "
                f"Relancez avec --force si c'est volontaire."
            )

    # --- 1. Suppression par source ---
    supprimees = 0
    for s, contenu in par_source.items():
        cur.execute(
            "DELETE FROM offres WHERE source = %s AND NOT (id = ANY(%s::text[]))",
            (s, list(contenu.keys())),
        )
        supprimees += cur.rowcount
    conn.commit()
    print(f"{supprimees} offre(s) supprimée(s) (disparues de l'export, sources concernées uniquement).")

    # --- 2. Nouvelles ou modifiées ---
    ids_actuels = [o["id"] for o in offres]
    cur.execute("SELECT id, texte_embedding FROM offres WHERE id = ANY(%s::text[])", (ids_actuels,))
    textes_existants = dict(cur.fetchall())

    a_traiter = []
    for o in offres:
        texte = build_texte_embedding(o)
        if not texte.strip():
            continue
        if textes_existants.get(o["id"]) != texte:
            a_traiter.append((o, texte))

    print(f"{len(a_traiter)} offre(s) nouvelle(s) ou modifiée(s) à (ré)indexer.")

    if a_traiter:
        print(f"Chargement du modèle {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        textes = [t for _, t in a_traiter]
        embeddings = model.encode(textes, show_progress_bar=True, batch_size=32)

        # --- 3. Upsert ---
        for (o, texte), emb in zip(a_traiter, embeddings):
            cur.execute(UPSERT_SQL, row_db(o, texte) + (emb.tolist(),))
        conn.commit()

    cur.execute("SELECT source, COUNT(*) FROM offres GROUP BY source ORDER BY source")
    print("\nTotal en base par source :")
    for s, n in cur.fetchall():
        print(f"  {s} : {n}")

    cur.close()
    conn.close()
    print("Synchronisation terminée.")


if __name__ == "__main__":
    main()