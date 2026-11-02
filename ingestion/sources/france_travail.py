import os
import time
import requests
from ..schema import Offre

class FranceTravail:
    name = "france_travail"
    TOKEN_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
    SEARCH_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
    SCOPE = "api_offresdemploiv2 o2dsoffre"
    PAGE = 150
    MAX_INDEX = 3149

    def _token(self) -> str:
        r = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": os.environ["FT_CLIENT_ID"],
                "client_secret": os.environ["FT_CLIENT_SECRET"],
                "scope": self.SCOPE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["access_token"]

    def fetch(self, mots_cles: list[str]):
        headers = {"Authorization": f"Bearer {self._token()}"}
        for mc in mots_cles:
            start = 0
            while start <= self.MAX_INDEX:
                end = min(start + self.PAGE - 1, self.MAX_INDEX)
                r = requests.get(self.SEARCH_URL, headers=headers,
                                 params={"motsCles": mc, "range": f"{start}-{end}"}, timeout=30)
                if r.status_code not in (200, 206):   # 204 = aucun résultat
                    break
                resultats = r.json().get("resultats", [])
                for o in resultats:
                    yield self._normalise(o)
                if len(resultats) < self.PAGE:
                    break
                start += self.PAGE
                time.sleep(0.2)
            time.sleep(0.2)

    @staticmethod
    def _normalise(o: dict) -> Offre:
        lieu = o.get("lieuTravail") or {}
        ent = o.get("entreprise") or {}
        return Offre(
            id=o['id'],
            source="france_travail",
            intitule=o.get("intitule") or "",
            entreprise=ent.get("nom"),
            lieu=lieu.get("libelle"),
            code_postal=lieu.get("codePostal"),
            type_contrat=o.get("typeContrat"),
            date_publication=o.get("dateCreation"),
            url=(o.get("origineOffre") or {}).get("urlOrigine"),  # champ à vérifier
            description=o.get("description") or "",
        )