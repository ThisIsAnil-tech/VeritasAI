import sys
from pathlib import Path

# Add src folder to PYTHONPATH
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import json
import pytest
from utils.config import get_app_config, AppConfig
from core.evaluator import Evaluator

@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return get_app_config()

@pytest.fixture(scope="session")
def evaluator(app_config) -> Evaluator:
    # Use mock model by default for unit tests to keep them offline and fast
    app_config.llm.default_model = "mock"
    return Evaluator(app_config)

def load_test_cases_by_category(category: str):
    """Utility to load matching category cases from golden_dataset.json."""
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "datasets" / "golden_dataset.json"
    if not dataset_path.exists():
        return []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [tc for tc in data.get("test_cases", []) if tc.get("category") == category]
    except Exception:
        return []
