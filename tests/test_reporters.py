import sys
import json
from pathlib import Path

# Add src folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from core.evaluator import EvaluationRunSummary, TestCaseResult
from core.metrics import MetricResult
from reporters.json_reporter import export_to_json
from reporters.html_reporter import generate_html_report

@pytest.fixture
def dummy_run_summary():
    metric_res = MetricResult(
        metric_name="relevance_score",
        score=0.9,
        passed=True,
        threshold=0.7,
        details={"match": True}
    )
    
    test_case_result = TestCaseResult(
        test_case_id="TC_DUMMY",
        category="relevance",
        prompt="Explain weather.",
        context="Weather details.",
        response_text="This is a response summarizing weather details.",
        expected_response="Weather details.",
        metrics={"relevance_score": metric_res},
        passed=True,
        latency=1.2,
        tokens=150,
        cost=0.0001,
        model="mock"
    )
    
    return EvaluationRunSummary(
        run_id="run_dummy_123",
        timestamp="2026-06-18 12:00:00",
        model="mock",
        total_test_cases=1,
        passed_count=1,
        failed_count=0,
        pass_rate=1.0,
        total_latency=1.2,
        avg_latency=1.2,
        total_tokens=150,
        total_cost=0.0001,
        metric_averages={"relevance_score": 0.9},
        results=[test_case_result]
    )

def test_json_reporter(dummy_run_summary, tmp_path):
    output_file = tmp_path / "report.json"
    written_path = export_to_json(dummy_run_summary, str(output_file))
    
    assert Path(written_path).exists()
    
    with open(written_path, "r") as f:
        data = json.load(f)
        
    assert data["run_id"] == "run_dummy_123"
    assert data["pass_rate"] == 1.0
    assert len(data["results"]) == 1

def test_html_reporter(dummy_run_summary, tmp_path):
    output_file = tmp_path / "report.html"
    written_path = generate_html_report(dummy_run_summary, str(output_file))
    
    assert Path(written_path).exists()
    
    with open(written_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "AI Quality Evaluation" in content
    assert "run_dummy_123" in content
    assert "relevance_score" in content
