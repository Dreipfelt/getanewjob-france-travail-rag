from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Offre:
    id: str
    source: str
    intitule: str
    entreprise: Optional[str] = None
    lieu: Optional[str] = None
    code_postal: Optional[str] = None
    region: Optional[str] = None
    type_contrat: Optional[str] = None
    date_publication: Optional[str] = None
    url: Optional[str] = None
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)