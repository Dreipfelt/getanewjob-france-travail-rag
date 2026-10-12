import re
import json
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime, date

CSP_CSV_URL = "https://www.data.gouv.fr/api/1/datasets/r/867034a2-2fa1-41b4-bd39-c84691ea618f"

MOTS_CLES_TITRE = [
    "data engineer", "data scientist", "machine learning", "mlops",
    "intelligence artificielle", "data analyst", "ingénieur données",
    "scientifique des données", "ingénieur ia", "analyste de données",
    "ingénieur en ia", "fouille de données", "traitement de données",
]

SENIOR_TITLE_WORDS = [
    "senior", "lead", "expert", "principal", "staff", "head of",
    "directeur", "responsable", "manager", "architecte", "chef de",
]

EXPERIENCE_BLOQUANTES = ["expert", "confirmé"]


def parse_date_fr(s) -> date:
    try:
        return datetime.strptime(str(s).strip(), "%d/%m/%Y").date()
    except Exception:
        return None


def is_active(row: pd.Series) -> bool:
    fin = parse_date_fr(row.get("Date de fin de publication par défaut"))
    if fin is None:
        return True
    return fin >= date.today()


def is_relevant(row: pd.Series) -> bool:
    title = str(row.get("Intitulé du poste", "") or "").lower()
    return any(m in title for m in MOTS_CLES_TITRE)


def is_junior_friendly(row: pd.Series) -> bool:
    title = str(row.get("Intitulé du poste", "")).lower()
    exp = str(row.get("Niveau d'expérience min. requis", "") or "").lower().strip()
    if any(w in title for w in SENIOR_TITLE_WORDS):
        return False
    if exp in EXPERIENCE_BLOQUANTES:
        return False
    return True


def normalise_row(row: pd.Series) -> dict:
    ref = row.get("Référence") or ""
    return {
        "id": ref,
        "source": "csp",
        "intitule": row.get("Intitulé du poste"),
        "metier": row.get("Métier"),
        "organisme": row.get("Organisme de rattachement"),
        "employeur": row.get("Employeur") or row.get("Organisme de rattachement"),
        "lieu": row.get("Lieu d'affectation"),
        "localisation": row.get("Localisation du poste"),
        "niveauEtudes": row.get("Niveau d'études"),
        "experienceRequise": row.get("Niveau d'expérience min. requis"),
        "natureEmploi": row.get("Nature de l'emploi"),
        "statut": row.get("Statut du poste"),
        "teletravail": str(row.get("Télétravail") or "").strip(),
        "datePublication": row.get("Date de première publication"),
        "dateFinPublication": row.get("Date de fin de publication par défaut"),
        "url": f"https://choisirleservicepublic.gouv.fr/offre-emploi/{ref}/detail",
    }


def ingest_csp(output_path: str = "csp_offres.json"):
    print("Téléchargement du CSV CSP...")
    resp = requests.get(CSP_CSV_URL, timeout=120)
    resp.raise_for_status()

    content = resp.content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(content), sep=";", dtype=str)
    print(f"  → {len(df)} offres brutes chargées")

    df = df[df.apply(is_active, axis=1)]
    print(f"  → {len(df)} offres encore actives")

    df = df[df.apply(is_relevant, axis=1)]
    print(f"  → {len(df)} offres après filtre mots-clés")

    df = df[df.apply(is_junior_friendly, axis=1)]
    print(f"  → {len(df)} offres après filtre junior/confirmé")

    results = [normalise_row(row) for _, row in df.iterrows()]

    out = Path(output_path)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Écrit dans {out.resolve()}")
    print(f"\nTop offres :")
    for o in results[:10]:
        print(f"  [{o['id']}] {o['intitule']} — {o['localisation']} ({o['experienceRequise'] or 'exp. non précisée'})")


if __name__ == "__main__":
    ingest_csp()
