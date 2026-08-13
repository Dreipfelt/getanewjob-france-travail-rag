"""
Ingestion consolidée des offres France Travail - API Offres d'emploi v2.

Ce script :
1. S'authentifie (OAuth2 client_credentials)
2. Interroge l'API séparément pour chaque mot-clé/expression de la liste
   MOTS_CLES (motsCles fonctionne en ET logique en interne, donc on ne peut
   pas combiner "data engineer" et "ML engineer" dans un seul appel sans
   forcer une recherche sur les deux à la fois - d'où les appels séparés)
3. Gère la pagination (max 150 résultats par appel, range jusqu'à 3149 max)
4. Dédoublonne les offres par leur identifiant unique (une offre peut
   apparaître dans plusieurs recherches, ex. "data scientist" et "ML engineer")
5. Exporte le résultat consolidé dans un fichier JSON horodaté

Limite connue et non contournée ici : si une recherche unique dépasse ~3150
résultats au global, l'API ne les restituera pas tous via ce seul filtre.
Non traité dans ce script - à surveiller si le volume par mot-clé s'avère
élevé (peu probable pour des intitulés de poste tech spécifiques, mais pas
vérifié empiriquement).

Prérequis :
    pip install requests python-dotenv

Fichier .env attendu (racine du projet, résolu automatiquement) :
    FT_CLIENT_ID=votre_identifiant_client
    FT_CLIENT_SECRET=votre_cle_secrete
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import requests

# Racine du projet, résolue par rapport à l'emplacement de ce fichier
# (ingestion/ingest_offres.py -> remonte d'un niveau). Fonctionne peu
# importe le répertoire depuis lequel le script est lancé.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"

load_dotenv(PROJECT_ROOT / ".env")

CLIENT_ID = os.getenv("FT_CLIENT_ID")
CLIENT_SECRET = os.getenv("FT_CLIENT_SECRET")

TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
SCOPE = "api_offresdemploiv2 o2dsoffre"
SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"

# Périmètre validé avec l'utilisateur. Modifiable ici directement.
MOTS_CLES = [
    "data engineer",
    "ML engineer",
    "machine learning engineer",
    "data scientist",
    "MLOps",
    "AI engineer",
    "data analyst",
]

PAGE_SIZE = 150          # Max autorisé par l'API
MAX_RANGE_INDEX = 3149   # Limite dure documentée par l'API (toutes requêtes confondues)
DELAY_BETWEEN_CALLS = 0.2  # 5 appels/seconde, marge de sécurité sous la limite de 10/s


def get_access_token() -> str:
    if not CLIENT_ID or not CLIENT_SECRET:
        print("ERREUR : FT_CLIENT_ID ou FT_CLIENT_SECRET manquant dans .env")
        sys.exit(1)

    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "scope": SCOPE,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    response = requests.post(TOKEN_URL, data=payload, headers=headers)

    if response.status_code != 200:
        print(f"Échec de l'authentification (status {response.status_code})")
        print(response.text)
        sys.exit(1)

    return response.json()["access_token"]


def search_offres_paginated(token: str, mots_cles: str) -> list:
    """
    Récupère TOUTES les offres pour un mot-clé donné, en paginant par blocs
    de PAGE_SIZE, jusqu'à épuisement des résultats ou jusqu'à MAX_RANGE_INDEX.
    """
    headers = {"Authorization": f"Bearer {token}"}
    all_offres = []
    start = 0

    while start <= MAX_RANGE_INDEX:
        end = min(start + PAGE_SIZE - 1, MAX_RANGE_INDEX)
        range_str = f"{start}-{end}"

        params = {
            "motsCles": mots_cles,
            "range": range_str,
        }

        response = requests.get(SEARCH_URL, headers=headers, params=params)

        # 204 = aucun résultat pour cette requête (page vide, fin de pagination)
        if response.status_code == 204:
            break

        if response.status_code not in (200, 206):
            print(f"  Erreur sur '{mots_cles}' (range {range_str}) : status {response.status_code}")
            print(f"  {response.text}")
            break

        data = response.json()
        resultats = data.get("resultats", [])
        all_offres.extend(resultats)

        # Si on reçoit moins que PAGE_SIZE résultats, on a atteint la fin
        if len(resultats) < PAGE_SIZE:
            break

        start += PAGE_SIZE
        time.sleep(DELAY_BETWEEN_CALLS)

    return all_offres


def main():
    token = get_access_token()
    print("Authentification OK.\n")

    offres_par_id = {}  # dédoublonnage par identifiant unique de l'offre

    for mot_cle in MOTS_CLES:
        print(f"Recherche : '{mot_cle}'...")
        offres = search_offres_paginated(token, mot_cle)
        print(f"  -> {len(offres)} offres trouvées")

        for offre in offres:
            offre_id = offre.get("id")
            if offre_id and offre_id not in offres_par_id:
                offres_par_id[offre_id] = offre

        time.sleep(DELAY_BETWEEN_CALLS)

    total_uniques = len(offres_par_id)
    print(f"\nTotal après dédoublonnage : {total_uniques} offres uniques")

    DATA_RAW_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DATA_RAW_DIR / f"offres_{timestamp}.json"

    output_data = {
        "date_extraction": datetime.now().isoformat(),
        "mots_cles_utilises": MOTS_CLES,
        "nombre_offres": total_uniques,
        "offres": list(offres_par_id.values()),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Fichier écrit : {output_path}")

    # Fichier "pointeur" vers le dernier run, sans timestamp dans le nom.
    # Permet à sync_db.py de toujours trouver le dernier fichier sans
    # avoir à parser des noms horodatés.
    latest_path = DATA_RAW_DIR / "offres_latest.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Pointeur mis à jour : {latest_path}")
    return output_path


if __name__ == "__main__":
    main()
