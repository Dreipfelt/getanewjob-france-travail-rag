import json
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from .sources.france_travail import FranceTravail
from .sources.csp import Csp
from .sources.forem import Forem

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Périmètre commun : à ajuster ici, une seule fois
MOTS_CLES = [
    "data engineer", "data scientist", "machine learning", "mlops",
    "intelligence artificielle", "data analyst", "ai engineer",
    "ingénieur données", "ingénieur ia",
]

SENIOR = re.compile(
    r"\b(senior|lead|expert|principal|staff|head of|directeur|responsable"
    r"|manager|architecte|chef de)\b"
)
EXPERIENCE = [re.compile(p) for p in [
    r"\b([3-9]|1[0-9])\s*ans?\s+d['\u2019]exp[ée]rience",
    r"\bexp[ée]rience\s+(de\s+)?([3-9]|1[0-9])\s*ans?",
    r"\bminimum\s+([3-9]|1[0-9])\s*ans?",
    r"\bau\s+moins\s+([3-9]|1[0-9])\s*ans?",
]]

SOURCES = [FranceTravail(), Csp(), Forem()]

def garder(o) -> bool:
    if SENIOR.search(o.intitule.lower()):
        return False
    return not any(p.search(o.description.lower()) for p in EXPERIENCE)

def main():
    vues = {}
    for src in SOURCES:
        brutes = 0
        for o in src.fetch(MOTS_CLES):
            brutes += 1
            if o.id not in vues and garder(o):
                vues[o.id] = o
        print(f"{src.name} : {brutes} offres brutes")

    export = {
        "date_extraction": datetime.now().isoformat(),
        "nombre_offres": len(vues),
        "offres": [o.to_dict() for o in vues.values()],
    }
    dest = ROOT / "data_raw"
    dest.mkdir(exist_ok=True)
    (dest / "offres_latest.json").write_text(
        json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Total après filtre et dédoublonnage : {len(vues)}")

if __name__ == "__main__":
    main()