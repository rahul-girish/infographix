from typing import Optional
from pydantic import BaseModel, ConfigDict

class AssetOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    url: str
    fileName: str
    contentType: str
    size: int
