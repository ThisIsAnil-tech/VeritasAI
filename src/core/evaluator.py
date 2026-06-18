import os
import time
import json
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from loguru import logger

from core.llm_client import LLMClient, LLMResponse
from core.metrics import (
    MetricResult,
    HallucinationMetric,
    ContextRetentionMetric,
    ResponseConsistencyMetric,
    RelevanceMetric,
    CompletenessMetric
)
from utils.config import AppConfig

class TestCaseResult(BaseModel):
    __test__ = False
    test_case_id: str
    category: str
    prompt: str
    context: Optional[str] = None
    response_text: str
    expected_response: Optional[str] = None
    metrics: Dict[str, MetricResult]
    passed: bool
    latency: float
    tokens: int
    cost: float
    model: str
    domain: str = "general"
    difficulty: str = "medium"

class EvaluationRunSummary(BaseModel):
    run_id: str
    timestamp: str
    model: str
    total_test_cases: int
    passed_count: int
    failed_count: int
    pass_rate: float
    total_latency: float
    avg_latency: float
    total_tokens: int
    total_cost: float
    metric_averages: Dict[str, float]
    results: List[TestCaseResult]

class Evaluator:
    def __init__(self, config: AppConfig, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client or LLMClient(
            default_model=config.llm.default_model,
            openai_api_key=config.openai_api_key,
            anthropic_api_key=config.anthropic_api_key,
            ollama_api_base=config.ollama_api_base,
            timeout=config.testing.timeout,
            retry_attempts=config.testing.retry_attempts
        )
        
        # Resolve output folders
        self.output_dir = Path(config.reporting.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.output_dir / ".cache.json"
        
        # Load Cache
        self.cache = self._load_cache()
        
        # Instantiate Metrics using config values
        metrics_settings = config.metrics
        
        h_settings = metrics_settings.get("hallucination", {})
        cr_settings = metrics_settings.get("context_retention", {})
        c_settings = metrics_settings.get("consistency", {})
        r_settings = metrics_settings.get("relevance", {})
        comp_settings = metrics_settings.get("completeness", {})

        self.metrics = {
            "hallucination": HallucinationMetric(
                threshold=h_settings.get("threshold", 0.8),
                method=h_settings.get("method", "hybrid")
            ),
            "context_retention": ContextRetentionMetric(
                threshold=cr_settings.get("threshold", 0.75)
            ),
            "consistency": ResponseConsistencyMetric(
                threshold=c_settings.get("threshold", 0.85)
            ),
            "relevance": RelevanceMetric(
                threshold=r_settings.get("threshold", 0.7),
                keyword_weight=r_settings.get("keyword_weight", 0.3)
            ),
            "completeness": CompletenessMetric(
                threshold=comp_settings.get("threshold", 0.8),
                min_length=comp_settings.get("min_length", 10),
                max_length=comp_settings.get("max_length", 2000)
            )
        }

    def _load_cache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}. Initializing empty cache.")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")

    def _get_cache_key(self, prompt: str, model: str, system: Optional[str], temp: float, max_tokens: int) -> str:
        key_dict = {
            "prompt": prompt,
            "model": model,
            "system": system,
            "temp": temp,
            "max_tokens": max_tokens
        }
        serialized = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _get_llm_response(self, prompt: str, model: str, system: Optional[str], temp: float, max_tokens: int) -> LLMResponse:
        """Fetch LLM response with caching support."""
        cache_key = self._get_cache_key(prompt, model, system, temp, max_tokens)
        
        # Check cache
        if cache_key in self.cache:
            logger.info("LLM cache hit!")
            cached = self.cache[cache_key]
            # Convert back to LLMResponse
            return LLMResponse(
                text=cached["text"],
                model=cached["model"],
                prompt_tokens=cached["prompt_tokens"],
                completion_tokens=cached["completion_tokens"],
                total_tokens=cached["total_tokens"],
                cost=cached["cost"],
                latency=cached["latency"],
                provider=cached["provider"],
                finish_reason=cached.get("finish_reason")
            )

        # Call client
        response = self.llm_client.generate(
            prompt=prompt,
            model=model,
            system_prompt=system,
            temperature=temp,
            max_tokens=max_tokens
        )

        # Cache response
        self.cache[cache_key] = {
            "text": response.text,
            "model": response.model,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "cost": response.cost,
            "latency": response.latency,
            "provider": response.provider,
            "finish_reason": response.finish_reason
        }
        self._save_cache()

        return response

    def evaluate_test_case(self, test_case: Dict[str, Any], model: str) -> TestCaseResult:
        """Run and score a single test case."""
        tc_id = test_case.get("id", "TC_UNKNOWN")
        category = test_case.get("category", "relevance")
        prompt = test_case.get("prompt", "")
        context = test_case.get("context")
        expected_response = test_case.get("expected_response")
        expected_key_points = test_case.get("expected_key_points") or []
        domain = test_case.get("domain", "general")
        difficulty = test_case.get("difficulty", "medium")

        # LLM completion parameters
        system_prompt = "You are a helpful assistant. Be concise, direct, and factual."
        temp = 0.7
        max_tokens = 2000

        start_time = time.time()
        
        # 1. Main generation
        response = self._get_llm_response(prompt, model, system_prompt, temp, max_tokens)
        response_text = response.text
        
        # 2. Gather consistency runs if evaluated
        alternate_responses = []
        consistency_runs = self.config.metrics.get("consistency", {}).get("runs", 3)
        if category == "consistency" or "consistency" in test_case.get("expected_metrics", {}):
            # Run additional generations with temperature (N-1 runs)
            # Use unique prompts or index parameters to prevent cache duplication for consistency runs
            for idx in range(1, consistency_runs):
                # We add a subtle comment to prompt to force bypass the standard identical cache key, or we change temp
                alt_resp = self._get_llm_response(
                    prompt + f"\n<!-- consistency run {idx} -->",
                    model,
                    system_prompt,
                    temp,
                    max_tokens
                )
                alternate_responses.append(alt_resp.text)

        elapsed = time.time() - start_time

        # 3. Execute Metrics
        metric_results = {}
        passed = True

        # Run all metrics configured in expected_metrics or defaults
        target_metrics = test_case.get("expected_metrics", {})
        if not target_metrics:
            # If not defined, run all metrics
            target_metrics = {k: self.metrics[k].threshold for k in self.metrics.keys()}

        for m_name, custom_threshold in target_metrics.items():
            if m_name in self.metrics:
                metric = self.metrics[m_name]
                # Override threshold if custom defined in test case
                original_threshold = metric.threshold
                metric.threshold = custom_threshold

                # Run evaluation
                try:
                    res = metric.evaluate(
                        response=response_text,
                        prompt=prompt,
                        context=context,
                        expected_response=expected_response,
                        expected_key_points=expected_key_points,
                        alternate_responses=alternate_responses
                    )
                    metric_results[m_name] = res
                    if not res.passed:
                        passed = False
                except Exception as e:
                    logger.error(f"Failed to evaluate metric {m_name} for {tc_id}: {str(e)}")
                    # Standard fail case
                    metric_results[m_name] = MetricResult(
                        metric_name=m_name,
                        score=0.0,
                        passed=False,
                        threshold=custom_threshold,
                        details={"error": str(e)}
                    )
                    passed = False
                finally:
                    # Restore threshold
                    metric.threshold = original_threshold

        return TestCaseResult(
            test_case_id=tc_id,
            category=category,
            prompt=prompt,
            context=context,
            response_text=response_text,
            expected_response=expected_response,
            metrics=metric_results,
            passed=passed,
            latency=response.latency,
            tokens=response.total_tokens,
            cost=response.cost,
            model=model,
            domain=domain,
            difficulty=difficulty
        )

    def run_suite(self, dataset_path: str, model: str) -> EvaluationRunSummary:
        """Run evaluation suite in parallel."""
        # Load dataset
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        test_cases = dataset.get("test_cases", [])
        if not test_cases:
            raise ValueError(f"No test cases found in dataset: {dataset_path}")

        run_id = f"run_{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting run {run_id} on {len(test_cases)} test cases using model: {model}")

        results = []
        workers = self.config.testing.parallel_workers

        # ThreadPool for parallel evaluations
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.evaluate_test_case, tc, model): tc
                for tc in test_cases
            }

            for fut in as_completed(futures):
                tc = futures[fut]
                tc_id = tc.get("id")
                try:
                    res = fut.result()
                    results.append(res)
                    status = "PASSED" if res.passed else "FAILED"
                    logger.info(f"Test case {tc_id} finished: {status}")
                except Exception as e:
                    logger.error(f"Exception during test case {tc_id} execution: {str(e)}")

        # Calculate aggregations
        total_tc = len(results)
        passed_count = sum(1 for r in results if r.passed)
        failed_count = total_tc - passed_count
        pass_rate = (passed_count / total_tc) if total_tc > 0 else 0.0
        
        total_latency = sum(r.latency for r in results)
        avg_latency = (total_latency / total_tc) if total_tc > 0 else 0.0
        total_tokens = sum(r.tokens for r in results)
        total_cost = sum(r.cost for r in results)

        # Average metric scores
        metric_sums = {}
        metric_counts = {}
        for r in results:
            for m_name, m_res in r.metrics.items():
                metric_sums[m_name] = metric_sums.get(m_name, 0.0) + m_res.score
                metric_counts[m_name] = metric_counts.get(m_name, 0) + 1

        metric_averages = {}
        for m_name, total_score in metric_sums.items():
            count = metric_counts[m_name]
            metric_averages[m_name] = round(total_score / count, 3)

        # Sort results by test case ID
        results.sort(key=lambda x: x.test_case_id)

        summary = EvaluationRunSummary(
            run_id=run_id,
            timestamp=timestamp,
            model=model,
            total_test_cases=total_tc,
            passed_count=passed_count,
            failed_count=failed_count,
            pass_rate=round(pass_rate, 4),
            total_latency=round(total_latency, 3),
            avg_latency=round(avg_latency, 3),
            total_tokens=total_tokens,
            total_cost=round(total_cost, 6),
            metric_averages=metric_averages,
            results=results
        )

        return summary
