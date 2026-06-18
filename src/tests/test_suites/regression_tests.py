import yaml
import pytest
from pathlib import Path
from tests.conftest import load_test_cases_by_category

# Load prompt files
def load_prompts_file(filepath: Path):
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

project_root = Path(__file__).resolve().parent.parent.parent.parent
v1_path = project_root / "src" / "data" / "prompts" / "v1_prompts.yaml"
v2_path = project_root / "src" / "data" / "prompts" / "v2_prompts.yaml"

prompts_v1 = load_prompts_file(v1_path).get("prompts", {})
prompts_v2 = load_prompts_file(v2_path).get("prompts", {})

# Use relevance test cases for demonstration
relevance_cases = load_test_cases_by_category("relevance")

@pytest.mark.regression
@pytest.mark.parametrize("test_case", relevance_cases, ids=lambda tc: tc["id"])
def test_prompt_regression(evaluator, test_case):
    """
    Evaluate prompt version v1 vs v2 to ensure no performance degradation.
    """
    model = "mock"
    tc_category = test_case.get("domain", "default")
    
    # 1. Fetch prompts for category
    p1 = prompts_v1.get(tc_category, prompts_v1.get("default", {}))
    p2 = prompts_v2.get(tc_category, prompts_v2.get("default", {}))
    
    # 2. Format user prompt with context if present
    context = test_case.get("context", "")
    prompt_text = test_case.get("prompt", "")
    
    # Format prompts
    user_p1 = p1.get("user", "{prompt}").format(prompt=prompt_text, context=context)
    user_p2 = p2.get("user", "{prompt}").format(prompt=prompt_text, context=context)
    
    # 3. Clone test case for each prompt
    tc_v1 = test_case.copy()
    tc_v1["prompt"] = user_p1
    
    tc_v2 = test_case.copy()
    tc_v2["prompt"] = user_p2
    
    # 4. Evaluate both
    res_v1 = evaluator.evaluate_test_case(tc_v1, model)
    res_v2 = evaluator.evaluate_test_case(tc_v2, model)
    
    # Compare relevance scores
    v1_score = res_v1.metrics.get("relevance_score").score
    v2_score = res_v2.metrics.get("relevance_score").score
    
    # Assert that version 2 does not degrade more than 0.05 margin compared to version 1
    margin = 0.05
    assert v2_score >= (v1_score - margin), (
        f"Regression detected for {test_case['id']}! "
        f"V1 Relevance Score: {v1_score}, V2 Relevance Score: {v2_score}"
    )
