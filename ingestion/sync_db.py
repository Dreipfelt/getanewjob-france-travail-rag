"""
Synchronisation base de données <-> dernier export France Travail.

Contrairement à load_embeddings.py (INSERT ... ON CONFLICT DO NOTHING, qui
n'update ni ne supprime rien), ce script :
1. Met à jour (upsert) les offres présentes dans le dernier export
2. Recalcule les embeddings pour les offres nouvelles ou modifiées
   (détection par comparaison du texte source, pas juste "id déjà présent")
3. Supprime de la base les offres qui ne sont plus dans l'export

Ce comportement est requis par l'article 5.2 de la licence de réutilisation
France Travail : "le Contenu créé, supprimé ou modifié de la Base de données
est respectivement créé, supprimé ou modifié de la Création."

Conçu pour être appelé par un DAG Airflow après ingest_offres.py.

Usage :
    python sync_db.py

Lit data_raw/offres_latest.json (généré par ingest_offres.py).
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"

load_dotenv(PROJECT_ROOT / ".env")

PG_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DB", "getanewjob"),
    "user": os.getenv("PG_USER", "getanewjob"),
    "password": os.getenv("PG_PASSWORD"),
}

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
INPUT_JSON = DATA_RAW_DIR / "offres_latest.json"


def build_texte_embedding(offre: dict) -> str:
    intitule = offre.get("intitule") or ""
    description = offre.get("description") or ""
    competences = offre.get("competences") or []
    libelles_competences = [c.get("libelle", "") for c in competences if c.get("libelle")]
    texte_competences = ". ".join(libelles_competences)
    parties = [p for p in [intitule, description, texte_competences] if p]
    return "\n\n".join(parties)


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"ERREUR : fichier introuvable : {INPUT_JSON}")
        print("Lancez d'abord ingest_offres.py.")
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    offres_actuelles = data.get("offres", [])
    print(f"{len(offres_actuelles)} offres dans le dernier export.\n")

    if not PG_CONFIG["password"]:
        print("ERREUR : PG_PASSWORD manquant dans .env")
        sys.exit(1)

    conn = psycopg2.connect(**PG_CONFIG)
    register_vector(conn)
    cur = conn.cursor()

    # --- 1. Suppression des offres disparues ---
    ids_actuels = {o["id"] for o in offres_actuelles if o.get("id")}

    cur.execute("SELECT id FROM offres")
    ids_en_base = {row[0] for row in cur.fetchall()}

    ids_a_supprimer = ids_en_base - ids_actuels
    if ids_a_supprimer:
        cur.execute("DELETE FROM offres WHERE id = ANY(%s)", [list(ids_a_supprimer)])
        print(f"{len(ids_a_supprimer)} offre(s) supprimée(s) (disparues de l'API).")
    else:
        print("Aucune offre à supprimer.")
    conn.commit()

    # --- 2. Détection des offres nouvelles ou modifiées ---
    # On compare le texte_embedding déjà stocké au texte recalculé : si
    # différent (ou absent), l'offre doit être (ré)insérée avec un nouvel
    # embedding. Évite de recalculer inutilement des embeddings identiques.
    cur.execute("SELECT id, texte_embedding FROM offres WHERE id = ANY(%s)", [list(ids_actuels)])
    textes_existants = dict(cur.fetchall())

    offres_a_traiter = []
    for offre in offres_actuelles:
        offre_id = offre.get("id")
        if not offre_id:
            continue
        texte = build_texte_embedding(offre)
        if not texte.strip():
            continue
        if textes_existants.get(offre_id) != texte:
            offres_a_traiter.append((offre, texte))

    print(f"{len(offres_a_traiter)} offre(s) nouvelle(s) ou modifiée(s) à (ré)indexer.")

    if not offres_a_traiter:
        cur.close()
        conn.close()
        print("Synchronisation terminée, aucun embedding à recalculer.")
        return

    print(f"\nChargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    textes = [t for _, t in offres_a_traiter]
    print(f"Calcul des embeddings pour {len(textes)} offre(s)...")
    embeddings = model.encode(textes, show_progress_bar=True, batch_size=32)

    print("Upsert en base...")
    for (offre, texte), embedding in zip(offres_a_traiter, embeddings):
        lieu = offre.get("lieuTravail") or {}
        entreprise = offre.get("entreprise") or {}

        cur.execute("""
            INSERT INTO offres (
                id, intitule, description, entreprise_nom,
                lieu_libelle, lieu_code_postal, lieu_latitude, lieu_longitude,
                type_contrat, type_contrat_libelle,
                experience_exige, experience_libelle,
                rome_code, rome_libelle,
                qualification_code, qualification_libelle,
                date_creation, date_actualisation,
                competences_json, salaire_json, contact_json, raw_json,
                texte_embedding, embedding
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (id) DO UPDATE SET
                intitule = EXCLUDED.intitule,
                description = EXCLUDED.description,
                entreprise_nom = EXCLUDED.entreprise_nom,
                lieu_libelle = EXCLUDED.lieu_libelle,
                lieu_code_postal = EXCLUDED.lieu_code_postal,
                lieu_latitude = EXCLUDED.lieu_latitude,
                lieu_longitude = EXCLUDED.lieu_longitude,
                type_contrat = EXCLUDED.type_contrat,
                type_contrat_libelle = EXCLUDED.type_contrat_libelle,
                experience_exige = EXCLUDED.experience_exige,
                experience_libelle = EXCLUDED.experience_libelle,
                rome_code = EXCLUDED.rome_code,
                rome_libelle = EXCLUDED.rome_libelle,
                qualification_code = EXCLUDED.qualification_code,
                qualification_libelle = EXCLUDED.qualification_libelle,
                date_creation = EXCLUDED.date_creation,
                date_actualisation = EXCLUDED.date_actualisation,
                competences_json = EXCLUDED.competences_json,
                salaire_json = EXCLUDED.salaire_json,
                contact_json = EXCLUDED.contact_json,
                raw_json = EXCLUDED.raw_json,
                texte_embedding = EXCLUDED.texte_embedding,
                embedding = EXCLUDED.embedding,
                date_extraction = now()
        """, (
            offre.get("id"),
            offre.get("intitule"),
            offre.get("description"),
            entreprise.get("nom"),
            lieu.get("libelle"),
            lieu.get("codePostal"),
            lieu.get("latitude"),
            lieu.get("longitude"),
            offre.get("typeContrat"),
            offre.get("typeContratLibelle"),
            offre.get("experienceExige"),
            offre.get("experienceLibelle"),
            offre.get("romeCode"),
            offre.get("romeLibelle"),
            offre.get("qualificationCode"),
            offre.get("qualificationLibelle"),
            offre.get("dateCreation"),
            offre.get("dateActualisation"),
            json.dumps(offre.get("competences"), ensure_ascii=False),
            json.dumps(offre.get("salaire"), ensure_ascii=False),
            json.dumps(offre.get("contact"), ensure_ascii=False),
            json.dumps(offre, ensure_ascii=False),
            texte,
            embedding.tolist(),
        ))

    conn.commit()

    cur.execute("SELECT COUNT(*) FROM offres")
    total_en_base = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"\nSynchronisation terminée. Total en base : {total_en_base}")


if __name__ == "__main__":
    main()
