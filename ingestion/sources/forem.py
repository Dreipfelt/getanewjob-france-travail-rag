import requests
from ..schema import Offre

class Forem:
    name = "forem"
    API_URL = "https://www.odwb.be/api/explore/v2.1/catalog/datasets/offres-d-emploi-forem/records"
    LIMIT = 100

    def fetch(self, mots_cles: list[str]):
        for mc in mots_cles:
            offset = 0
            while True:
                r = requests.get(self.API_URL,
                                 params={"where": f'"{mc}"', "limit": self.LIMIT, "offset": offset},
                                 timeout=30)
                if r.status_code != 200:
                    break
                data = r.json()
                recs = data.get("results", [])
                for x in recs:
                    yield Offre(
                        id=f"forem:{x.get('numerooffreforem')}",
                        source="forem",
                        intitule=x.get("titreoffre") or "",
                        entreprise=x.get("nomemployeur"),
                        lieu=", ".join(x.get("lieuxtravaillocalite") or []) or None,
                        code_postal=", ".join(x.get("lieuxtravailcodepostal") or []) or None,
                        region=", ".join(x.get("lieuxtravailregion") or []) or None,
                        type_contrat=x.get("typecontrat"),
                        date_publication=x.get("datedebutdiffusion"),
                        url=x.get("url"),
                        description=x.get("descriptionfonction") or x.get("description") or "",
                    )
                offset += self.LIMIT
                if not recs or offset >= data.get("total_count", 0):
                    break