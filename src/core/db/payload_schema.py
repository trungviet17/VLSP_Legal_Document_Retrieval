import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from pydantic import BaseModel, Field
from typing import List


class IDPayLoadSchema(BaseModel): 
    text: str = Field(..., description="The text content of the payload")
    child_id: List[str] = Field(..., description="List of child IDs associated with the payload")
    law_id: str = Field(..., description="The unique identifier for the law")


class AIDPayloadSchema(BaseModel): 
    id: str = Field(..., description="The unique identifier for the payload")
    aid: str = Field(..., description="The unique identifier for the article")
    law_id: str = Field(..., description="The unique identifier for the law")
    text: str = Field(..., description="The text content of the article")






