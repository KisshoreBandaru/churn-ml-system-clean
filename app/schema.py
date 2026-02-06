from typing import Literal
from pydantic import BaseModel

class ChurnInput(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    support_calls: int  
    contract_type: Literal["monthly", "yearly"]
