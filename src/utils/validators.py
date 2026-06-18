import json
from typing import Dict, Any, Tuple, List, Optional
from pydantic import BaseModel, Field, ValidationError

class TestCaseSchema(BaseModel):
    id: str
    category: str
    prompt: str
    context: Optional[str] = None
    expected_response: Optional[str] = None
    expected_metrics: Dict[str, float] = Field(default_factory=dict)
    domain: Optional[str] = "general"
    difficulty: Optional[str] = "medium"
    expected_key_points: Optional[List[str]] = None

class DatasetSchema(BaseModel):
    test_cases: List[TestCaseSchema]

def validate_dataset_json(filepath: str) -> Tuple[bool, List[str]]:
    """
    Validate a dataset file matching the DatasetSchema.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON file format: {str(e)}"]
    except Exception as e:
        return False, [f"Failed to read file: {str(e)}"]

    try:
        DatasetSchema(**data)
        return True, []
    except ValidationError as e:
        for err in e.errors():
            loc = " -> ".join(map(str, err["loc"]))
            msg = err["msg"]
            errors.append(f"Validation error at [{loc}]: {msg}")
        return False, errors
