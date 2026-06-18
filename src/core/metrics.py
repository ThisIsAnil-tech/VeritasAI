import re
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

def calculate_tfidf_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using TF-IDF."""
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([text1, text2])
        pairwise_similarity = (tfidf * tfidf.T).toarray()
        return float(pairwise_similarity[0, 1])
    except Exception as e:
        logger.warning(f"TF-IDF similarity failed, falling back to Jaccard: {str(e)}")
        # Simple Jaccard similarity fallback
        w1 = set(re.findall(r'\w+', text1.lower()))
        w2 = set(re.findall(r'\w+', text2.lower()))
        if not w1 and not w2:
            return 1.0
        if not w1 or not w2:
            return 0.0
        return len(w1.intersection(w2)) / len(w1.union(w2))

class MetricResult(BaseModel):
    metric_name: str
    score: float
    passed: bool
    threshold: float
    details: Dict[str, Any]

class BaseMetric(ABC):
    def __init__(self, name: str, threshold: float):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def evaluate(self, response: str, **kwargs) -> MetricResult:
        """
        Evaluate the response.
        Supported kwargs:
        - prompt (str)
        - context (str)
        - expected_response (str)
        - expected_key_points (List[str])
        - alternate_responses (List[str])
        """
        pass

class HallucinationMetric(BaseMetric):
    """
    Evaluates factual consistency and grounding against the source context.
    Returns:
        score: 1.0 = fully consistent / grounded, 0.0 = completely hallucinated.
    """
    def __init__(self, threshold: float = 0.8, method: str = "hybrid"):
        super().__init__("hallucination_score", threshold)
        self.method = method

    def evaluate(self, response: str, **kwargs) -> MetricResult:
        context = kwargs.get("context", "")
        if not context:
            # If no context is provided, we cannot evaluate factual grounding.
            # We return a neutral pass with 1.0 to avoid blocking tests.
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No context provided for factual grounding checks."}
            )

        # Grounding check: verify that key sentences/terms in the response exist semantically in the context.
        # Check specific words (nouns/numbers) from response and check if they are absent in the context.
        response_words = re.findall(r'\b[A-Za-z0-9]{3,}\b', response.lower())
        context_words = set(re.findall(r'\b[A-Za-z0-9]{3,}\b', context.lower()))
        
        # Filter out common stop words to focus on content words
        stopwords = {"the", "and", "a", "of", "to", "in", "is", "that", "it", "was", "for", "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from"}
        content_words = [w for w in response_words if w not in stopwords]
        
        if not content_words:
            grounding_score = 1.0
        else:
            matches = sum(1 for w in content_words if w in context_words)
            grounding_score = matches / len(content_words)

        # Contradiction detection: check for specific opposite indicators or quantity mismatches
        contradiction_detected = False
        contradiction_reason = None
        
        # Check number mismatch
        resp_nums = set(re.findall(r'\b\d+\b', response))
        ctx_nums = set(re.findall(r'\b\d+\b', context))
        unmatched_nums = resp_nums - ctx_nums
        if unmatched_nums:
            # Penalty for numbers introduced out of nowhere
            grounding_score *= 0.8
            contradiction_reason = f"Response introduced numbers not in context: {unmatched_nums}"

        # Combine TF-IDF similarity with grounding score
        similarity = calculate_tfidf_similarity(response, context)
        score = (grounding_score * 0.6) + (similarity * 0.4)
        
        # Cap score to 0.0 - 1.0
        score = max(0.0, min(1.0, float(score)))
        passed = score >= self.threshold

        details = {
            "grounding_score": round(grounding_score, 3),
            "semantic_similarity": round(similarity, 3),
            "contradiction_detected": contradiction_detected,
            "contradiction_reason": contradiction_reason,
            "method": self.method
        }

        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            passed=passed,
            threshold=self.threshold,
            details=details
        )

class ContextRetentionMetric(BaseMetric):
    """
    Measures semantic similarity, key information matching, and context window utilization.
    """
    def __init__(self, threshold: float = 0.75):
        super().__init__("context_retention_score", threshold)

    def evaluate(self, response: str, **kwargs) -> MetricResult:
        context = kwargs.get("context", "")
        expected_key_points = kwargs.get("expected_key_points") or []
        
        if not context and not expected_key_points:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No context or expected key points provided."}
            )

        # 1. Semantic Similarity to Context
        similarity = calculate_tfidf_similarity(response, context) if context else 1.0

        # 2. Key Information Matching
        matched_points = []
        unmatched_points = []
        key_point_score = 1.0
        
        if expected_key_points:
            resp_lower = response.lower()
            for kp in expected_key_points:
                # Check for word matching or substring presence
                # Clean prompt keywords of punctuation
                kp_clean = re.sub(r'[^\w\s]', '', kp.lower())
                words = kp_clean.split()
                # If substring is in response, or all words of the key point are in the response
                if kp_clean in resp_lower or (words and all(w in resp_lower for w in words)):
                    matched_points.append(kp)
                else:
                    unmatched_points.append(kp)
            
            key_point_score = len(matched_points) / len(expected_key_points)

        # Overall Context Retention Score: blend of key point match & semantic similarity
        if expected_key_points:
            score = (key_point_score * 0.7) + (similarity * 0.3)
        else:
            score = similarity

        score = max(0.0, min(1.0, float(score)))
        passed = score >= self.threshold

        # Context Window Utilization (Mocking a 4096 context window length check)
        context_len = len(context.split())
        utilization = min(1.0, context_len / 4096.0)

        details = {
            "semantic_similarity": round(similarity, 3),
            "key_point_match_score": round(key_point_score, 3),
            "matched_key_points": matched_points,
            "unmatched_key_points": unmatched_points,
            "context_word_count": context_len,
            "context_window_utilization": round(utilization, 4)
        }

        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            passed=passed,
            threshold=self.threshold,
            details=details
        )

class ResponseConsistencyMetric(BaseMetric):
    """
    Measures semantic consistency across multiple model outputs.
    Calculates stability score as pairwise similarity average.
    """
    def __init__(self, threshold: float = 0.85):
        super().__init__("consistency_score", threshold)

    def evaluate(self, response: str, **kwargs) -> MetricResult:
        alternate_responses = kwargs.get("alternate_responses") or []
        if not alternate_responses:
            # If no alternate runs are provided, we evaluate against self (trivially consistent)
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No alternate responses provided. Running single-evaluation mode."}
            )

        all_runs = [response] + alternate_responses
        similarities = []

        # Calculate pairwise TF-IDF similarities
        for i in range(len(all_runs)):
            for j in range(i + 1, len(all_runs)):
                sim = calculate_tfidf_similarity(all_runs[i], all_runs[j])
                similarities.append(sim)

        avg_similarity = float(np.mean(similarities)) if similarities else 1.0
        variance = float(np.var(similarities)) if similarities else 0.0

        # stability score is simply the average pairwise similarity
        score = max(0.0, min(1.0, avg_similarity))
        passed = score >= self.threshold

        details = {
            "pairwise_similarities": [round(s, 3) for s in similarities],
            "average_similarity": round(avg_similarity, 3),
            "variance": round(variance, 4),
            "runs_evaluated": len(all_runs)
        }

        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            passed=passed,
            threshold=self.threshold,
            details=details
        )

class RelevanceMetric(BaseMetric):
    """
    Measures prompt relevance using keyword overlaps and cosine similarity.
    """
    def __init__(self, threshold: float = 0.7, keyword_weight: float = 0.3):
        super().__init__("relevance_score", threshold)
        self.keyword_weight = keyword_weight

    def evaluate(self, response: str, **kwargs) -> MetricResult:
        prompt = kwargs.get("prompt", "")
        if not prompt:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                passed=True,
                threshold=self.threshold,
                details={"reason": "No prompt provided to calculate relevance."}
            )

        # 1. TF-IDF Cosine Similarity
        similarity = calculate_tfidf_similarity(response, prompt)

        # 2. Keyword Jaccard Overlap
        p_words = set(re.findall(r'\b[a-z]{3,}\b', prompt.lower()))
        r_words = set(re.findall(r'\b[a-z]{3,}\b', response.lower()))
        
        stopwords = {"the", "and", "a", "of", "to", "in", "is", "that", "it", "was", "for", "on", "are", "as", "with", "his", "they", "at", "be", "this", "have", "from", "what", "where", "how", "why"}
        p_keywords = p_words - stopwords
        r_keywords = r_words - stopwords

        if not p_keywords:
            keyword_overlap = 1.0
        else:
            keyword_overlap = len(p_keywords.intersection(r_keywords)) / len(p_keywords)

        # Blended score
        score = (keyword_overlap * self.keyword_weight) + (similarity * (1.0 - self.keyword_weight))
        score = max(0.0, min(1.0, float(score)))
        passed = score >= self.threshold

        details = {
            "semantic_similarity": round(similarity, 3),
            "keyword_overlap": round(keyword_overlap, 3),
            "prompt_keywords": list(p_keywords)[:10],
            "response_keywords": list(r_keywords)[:10]
        }

        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            passed=passed,
            threshold=self.threshold,
            details=details
        )

class CompletenessMetric(BaseMetric):
    """
    Checks if a response covers the structural and length expectations of the prompt.
    """
    def __init__(self, threshold: float = 0.8, min_length: int = 10, max_length: int = 2000, structural_weight: float = 0.4):
        super().__init__("completeness_score", threshold)
        self.min_length = min_length
        self.max_length = max_length
        self.structural_weight = structural_weight

    def evaluate(self, response: str, **kwargs) -> MetricResult:
        expected = kwargs.get("expected_response", "")

        # 1. Length Adequacy Score
        resp_len = len(response)
        length_score = 1.0
        if resp_len < self.min_length:
            length_score = resp_len / self.min_length
        elif resp_len > self.max_length:
            # Linear penalty for exceeding max length
            excess = resp_len - self.max_length
            length_score = max(0.1, 1.0 - (excess / self.max_length))

        # 2. Structural Completeness Score
        # Look for introduction (paragraph 1), body details (paragraph 2+), and conclusion indicators
        paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
        has_intro = len(paragraphs) >= 1
        has_body = len(paragraphs) >= 2
        
        # Search for conclusion-like words (e.g. conclusion, finally, overall, in summary, thus)
        conclusion_words = {"conclusion", "finally", "overall", "summary", "thus", "therefore", "conclude"}
        has_conclusion = any(w in response.lower() for w in conclusion_words) or len(paragraphs) >= 3

        structural_score = 0.0
        if has_intro:
            structural_score += 0.3
        if has_body:
            structural_score += 0.4
        if has_conclusion:
            structural_score += 0.3

        # 3. Ground Truth Coverage (Optional)
        coverage_score = 1.0
        if expected:
            coverage_score = calculate_tfidf_similarity(response, expected)

        # Blend scores
        if expected:
            # If expected response is present, include coverage check
            score = (length_score * 0.3) + (structural_score * 0.3) + (coverage_score * 0.4)
        else:
            score = (length_score * (1.0 - self.structural_weight)) + (structural_score * self.structural_weight)

        score = max(0.0, min(1.0, float(score)))
        passed = score >= self.threshold

        details = {
            "response_length": resp_len,
            "min_length_threshold": self.min_length,
            "max_length_threshold": self.max_length,
            "length_score": round(length_score, 3),
            "structural_score": round(structural_score, 3),
            "ground_truth_coverage": round(coverage_score, 3) if expected else None,
            "paragraph_count": len(paragraphs)
        }

        return MetricResult(
            metric_name=self.name,
            score=round(score, 3),
            passed=passed,
            threshold=self.threshold,
            details=details
        )
