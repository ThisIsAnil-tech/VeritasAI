import json
from pathlib import Path
from core.evaluator import EvaluationRunSummary
from loguru import logger

def export_to_json(summary: EvaluationRunSummary, output_path: str) -> str:
    """
    Exports the EvaluationRunSummary to a structured JSON file.
    Returns the path to the written file.
    """
    path = Path(output_path)
    # If path is a directory, write a default filename
    if path.is_dir() or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        path = path / f"report_{summary.run_id}.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Convert Pydantic object to dictionary, serializing custom types
        data = json.loads(summary.model_dump_json())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"JSON report written successfully to {path}")
        return str(path)
    except Exception as e:
        logger.error(f"Failed to write JSON report to {path}: {str(e)}")
        raise e
