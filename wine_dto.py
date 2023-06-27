from typing import List, Optional
from pydantic import BaseModel


class WineDTO(BaseModel):
    fixedAcidity: float
    volatileAcidity: float
    citricAcid: float
    residualSugar: float
    chlorides: float
    freeSulfurDioxide: float
    totalSulfurDioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    color: Optional[str]


class WinesDTO(BaseModel):
    wines: List[WineDTO]
