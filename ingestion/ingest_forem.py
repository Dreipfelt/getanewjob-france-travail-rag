import os
import re
import json
import requests
from pathlib import Path
from datetime import datetime

FOREM_API_URL = "https://www.odwb.be/api/explore/v2.1/catalog/datasets/offres-d-emploi-forem/records"

MOTS_CLES = [
    "data engineer",
    "data scientist",
    "machine learning",
    "MLOps",
    "intelligence artificielle",
    "data analyst",
]

SENIOR_TITLE_WORDS = [
    "senior", "lead", "expert", "principal", "staff",
    "head of", "directeur", "responsable", "manager", "architecte",
]

EXP_PATTERNS = [
    r"\b([3-9]|1[0-9])\s*ans?\s+d['\u2019]exp\u00e9rience",
    r"\bexp\u00e9rience\s+(de\s+)?([3-9]|1[0-9])\s*ans?",
    r"\bminimum\s+([3-9]|1[0-9])\s*ans?",
    r"\bau\s+moins\s+([3-9]|1[0-9])\s*ans?",
]


def is_junior_friendly(offre: dict) -> bool:
    title = (offre.get("titreoffre") or "").lower()
    description = (offre.get("descriptionfonction") or offre.get("description") or "").lower()

    if any(w in title for w in SENIOR_TITLE_WORDS):
        return False
    for pattern in EXP_PATTERNS:
        if re.search(pattern, description):
            return False
    return True


def normalise_offre(offre: dict) -> dict:
    return {
        "id": offre.get("numerooffreforem"),
        "source": "forem",
        "intitule": offre.get("titreoffre"),
        "entreprise": offre.get("nomemployeur"),
        "lieu": offre.get("lieuxtravaillocalite"),
        "region": offre.get("lieuxtravailregion"),
        "codePostal": offre.get("lieuxtravailcodepostal"),
        "typeContrat": offre.get("typecontrat"),
        "datePublication": offre.get("datedebutdiffusion"),
        "url": offre.get("url"),
        "description": offre.get("descriptionfonction") or offre.get("description") or "",
        "raw": offre,
    }


def fetch_forem(mot_cle: str, limit: int = 100) -> list[dict]:
    results = []
    offset = 0

    while True:
        params = {
            "where": f'"{mot_cle}"',  # ODSQL plain full-text search
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(FOREM_API_URL, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  Erreur {resp.status_code} pour '{mot_cle}': {resp.text[:200]}")
            break

        data = resp.json()
        records = data.get("results", [])
        results.extend(records)

        total = data.get("total_count", 0)
        offset += limit
        if offset >= total or not records:
            break

    return results


def ingest_forem(output_path: str = "forem_offres.json"):
    all_offres = []
    seen_ids = set()

    for mot_cle in MOTS_CLES:
        print(f"Recherche Forem : '{mot_cle}'")
        records = fetch_forem(mot_cle)
        print(f"  → {len(records)} offres brutes")

        for r in records:
            offre = normalise_offre(r)
            oid = offre["id"]
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            if is_junior_friendly(offre):
                all_offres.append(offre)

    print(f"\nTotal après dédoublonnage et filtre : {len(all_offres)} offres")

    out = Path(output_path)
    out.write_text(json.dumps(all_offres, ensure_ascii=False, indent=2))
    print(f"Écrit dans {out.resolve()}")


if __name__ == "__main__":
    ingest_forem()