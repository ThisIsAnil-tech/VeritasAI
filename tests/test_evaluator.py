import sys
import json
from pathlib import Path

# Add src folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from utils.config import get_app_config
from core.evaluator import Evaluator

@pytest.fixture
def evaluator_instance():
    config = get_app_config()
    config.llm.default_model = "mock"
    # Ensure cache uses a temp or local testing path
    config.reporting.output_dir = "./reports"
    return Evaluator(config)

def test_evaluator_init(evaluator_instance):
    assert evaluator_instance is not None
    assert "relevance" in evaluator_instance.metrics
    assert "hallucination" in evaluator_instance.metrics

def test_evaluate_single_case(evaluator_instance):
    test_case = {
        "id": "TC_TEST_01",
        "category": "relevance",
        "prompt": "What is the capital of France?",
        "context": "France is a country in Europe. Paris is the capital.",
        "expected_response": "Paris",
        "expected_metrics": {
            "relevance_score": 0.7
        }
    }
    
    result = evaluator_instance.evaluate_test_case(test_case, model="mock")
    assert result.test_case_id == "TC_TEST_01"
    assert result.passed
    assert result.response_text is not None
    assert "relevance_score" in result.metrics
    assert result.metrics["relevance_score"].passed

def test_evaluator_caching(evaluator_instance):
    # Clear cache first
    evaluator_instance.cache = {}
    evaluator_instance._save_cache()
    
    prompt = "Test caching functionality"
    model = "mock"
    
    # First fetch (cache miss)
    resp1 = evaluator_instance._get_llm_response(prompt, model, None, 0.7, 2000)
    cache_key = evaluator_instance._get_cache_key(prompt, model, None, 0.7, 2000)
    assert cache_key in evaluator_instance.cache
    
    # Second fetch (cache hit)
    resp2 = evaluator_instance._get_llm_response(prompt, model, None, 0.7, 2000)
    assert resp1.text == resp2.text
    # Latency of cached response should be from cache
    assert resp2.latency == resp1.latency

def test_run_suite(evaluator_instance, tmp_path):
    # Create a small dataset JSON file
    dataset = {
        "test_cases": [
            {
                "id": "T001",
                "category": "relevance",
                "prompt": "What is the capital of France?",
                "context": "France capital is Paris.",
                "expected_metrics": {"relevance_score": 0.6}
            },
            {
                "id": "T002",
                "category": "completeness",
                "prompt": "Summarize coding structure.",
                "context": "Write clean greet function in python.",
                "expected_metrics": {"completeness_score": 0.6}
            }
        ]
    }
    
    dataset_file = tmp_path / "test_suite.json"
    with open(dataset_file, "w") as f:
        json.dump(dataset, f)
        
    summary = evaluator_instance.run_suite(str(dataset_file), model="mock")
    
    assert summary.total_test_cases == 2
    assert summary.passed_count == 2
    assert len(summary.results) == 2
    assert summary.pass_rate == 1.0
    assert summary.total_tokens > 0
