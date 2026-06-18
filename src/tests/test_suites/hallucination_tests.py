import pytest
from tests.conftest import load_test_cases_by_category

test_cases = load_test_cases_by_category("hallucination")

@pytest.mark.hallucination
@pytest.mark.parametrize("test_case", test_cases, ids=lambda tc: tc["id"])
def test_hallucination_detection(evaluator, test_case):
    """
    Test fact-checking, source grounding, and contradiction detection.
    Runs the LLM and runs HallucinationMetric checking if score exceeds the threshold.
    """
    model = "mock"
    # Execute single test case evaluation
    result = evaluator.evaluate_test_case(test_case, model)
    
    # Assert main outcomes
    assert result.passed, f"Test case {test_case['id']} failed. Results: {result.metrics}"
    
    # Assert specific hallucination metric output
    h_metric = result.metrics.get("hallucination_score")
    assert h_metric is not None, "Hallucination score metric is missing"
    assert h_metric.score >= h_metric.threshold, (
        f"Score {h_metric.score} is below threshold {h_metric.threshold}. "
        f"Details: {h_metric.details}"
    )
