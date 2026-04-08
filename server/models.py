from pydantic import BaseModel, Field
from typing import Literal, List

class Observation(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the prompt.")
    prompt_preview: str = Field(..., description="The user's prompt text.")
    estimated_tokens: int = Field(..., description="Estimated token length.")
    complexity: Literal["low", "moderate", "high"] = Field(..., description="Semantic complexity.")
    is_code: bool = Field(..., description="True if the prompt contains programming/coding tasks.")
    system_load: Literal["low", "high"] = Field(..., description="Current server traffic. High load causes large models to lag.")
    previous_actions: List[str] = Field(..., description="History of models chosen.")

class Action(BaseModel):
    model_choice: Literal["use_small_model", "use_medium_model", "use_large_model"] = Field(
        ..., description="Which model to route to."
    )