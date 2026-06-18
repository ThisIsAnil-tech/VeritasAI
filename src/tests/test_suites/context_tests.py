import pytest
from tests.conftest import load_test_cases_by_category

test_cases = load_test_cases_by_category("context_retention")

@pytest.mark.context
@pytest.mark.parametrize("test_case", test_cases, ids=lambda tc: tc["id"])
def test_context_retention(evaluator, test_case):
    """
    Test context retention capabilities (long context and conversation tracking).
    Runs the LLM and runs ContextRetentionMetric.
    """
    model = "mock"
    result = evaluator.evaluate_test_case(test_case, model)
    
    assert result.passed, f"Context retention test failed. Results: {result.metrics}"
    
    cr_metric = result.metrics.get("context_retention_score")
    assert cr_metric is not None, "Context retention score metric is missing"
    assert cr_metric.score >= cr_metric.threshold, (
        f"Score {cr_metric.score} is below threshold {cr_metric.threshold}. "
        f"Details: {cr_metric.details}"
    )
