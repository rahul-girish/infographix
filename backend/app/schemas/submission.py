from typing import Optional, Any
from pydantic import BaseModel, ConfigDict

class SubmissionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    projectId: str
    status: str
    submittedAt: str
