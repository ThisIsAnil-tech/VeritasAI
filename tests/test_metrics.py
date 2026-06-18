import sys
from pathlib import Path

# Add src folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from core.metrics import (
    HallucinationMetric,
    ContextRetentionMetric,
    ResponseConsistencyMetric,
    RelevanceMetric,
    CompletenessMetric
)

def test_relevance_metric():
    metric = RelevanceMetric(threshold=0.7)
    # High relevance case
    res1 = metric.evaluate(
        response="Paris is the capital of France.",
        prompt="What is the capital of France?"
    )
    assert res1.passed
    assert res1.score > 0.7

    # Low relevance case
    res2 = metric.evaluate(
        response="The quick brown fox jumps over the lazy dog.",
        prompt="What is the capital of France?"
    )
    assert not res2.passed
    assert res2.score < 0.5

def test_hallucination_metric():
    metric = HallucinationMetric(threshold=0.8)
    
    # Grounded case
    res1 = metric.evaluate(
        response="William Shakespeare was a playwright.",
        context="William Shakespeare was an English playwright who wrote plays."
    )
    assert res1.passed
    assert res1.score >= 0.8

    # Ungrounded/Hallucinated case: contains names and terms not in context
    res2 = metric.evaluate(
        response="John Doe was a pilot who flew boeing planes.",
        context="William Shakespeare was an English playwright who wrote plays."
    )
    assert not res2.passed
    assert res2.score < 0.6

def test_context_retention_metric():
    metric = ContextRetentionMetric(threshold=0.75)
    
    # Good retention with key point matching
    res1 = metric.evaluate(
        response="The client wants to use HTTPS and JSON payloads.",
        context="All requests must use HTTPS and JSON payloads.",
        expected_key_points=["HTTPS", "JSON"]
    )
    assert res1.passed
    assert len(res1.details["matched_key_points"]) == 2

    # Poor retention (missing key points)
    res2 = metric.evaluate(
        response="We will send a response.",
        context="All requests must use HTTPS and JSON payloads.",
        expected_key_points=["HTTPS", "JSON"]
    )
    assert not res2.passed

def test_completeness_metric():
    metric = CompletenessMetric(threshold=0.8, min_length=10, max_length=200)
    
    # Complete text matching length and structure bounds
    res1 = metric.evaluate(
        response="Introduction: The process is simple. Body details: We connect and verify. Conclusion: This completes the workflow.",
    )
    assert res1.passed
    assert res1.details["length_score"] == 1.0
    assert res1.details["structural_score"] == 1.0

    # Incomplete (too short)
    res2 = metric.evaluate(
        response="Short.",
    )
    assert not res2.passed

def test_consistency_metric():
    metric = ResponseConsistencyMetric(threshold=0.85)
    
    # High consistency
    res1 = metric.evaluate(
        response="Paris is the capital.",
        alternate_responses=["Paris is the capital of France.", "Paris is the main capital."]
    )
    assert res1.passed
    assert res1.score >= 0.85

    # Low consistency
    res2 = metric.evaluate(
        response="Paris is the capital.",
        alternate_responses=["London is the capital.", "Tokyo is the capital."]
    )
    assert not res2.passed
    assert res2.score < 0.7
