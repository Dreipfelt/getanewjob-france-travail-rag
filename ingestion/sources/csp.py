from datetime import datetime, date
from io import StringIO
import requests
import pandas as pd
from ..schema import Offre

class Csp:
    name = "csp"
    CSV_URL = "https://www.data.gouv.fr/api/1/datasets/r/867034a2-2fa1-41b4-bd39-c84691ea618f"

    def fetch(self, mots_cles: list[str]):
        r = requests.get(self.CSV_URL, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.content.decode("utf-8", errors="replace")),
                         sep=";", dtype=str).fillna("")
        for _, row in df.iterrows():
            titre = row.get("Intitulé du poste", "")
            if not any(m in titre.lower() for m in mots_cles):
                continue
            try:
                if datetime.strptime(row.get("Date de fin de publication par défaut", ""),
                                     "%d/%m/%Y").date() < date.today():
                    continue   # offre expirée
            except ValueError:
                pass           # date absente ou invalide : on garde
            ref = row.get("Référence", "")
            niveau = str(row.get("Niveau d'expérience min. requis", "")).lower().strip()
            if niveau in ("expert", "confirmé"):
                continue
            yield Offre(
                id=f"csp:{ref}",
                source="csp",
                intitule=titre,
                entreprise=row.get("Employeur") or row.get("Organisme de rattachement") or None,
                lieu=row.get("Lieu d'affectation") or None,
                type_contrat=row.get("Nature de l'emploi") or None,
                date_publication=row.get("Date de première publication") or None,
                url=f"https://choisirleservicepublic.gouv.fr/offre-emploi/{ref}/detail",
                description="",   # absent du CSV : à compléter via fetch_csp_details.py
            )