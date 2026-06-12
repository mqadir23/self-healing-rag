"""
evaluator.py — Hybrid evaluation module for the Self-Healing RAG pipeline.

Combines fast, rule-based heuristic checks with a deep LLM-as-a-judge (Groq/Llama-3-70b)
evaluation step. Computes standard RAG metrics (faithfulness, answer relevance,
context relevance, completeness) to determine pass/fail status and diagnose failure modes.
"""

import os
import json
import re
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

# Industry-standard weights for computed overall score (faithfulness weighted highest)
METRIC_WEIGHTS = {
    "faithfulness": 0.35,
    "answer_relevance": 0.30,
    "context_relevance": 0.20,
    "answer_completeness": 0.15,
}

EVAL_PROMPT = """You are an evaluation system. Score this RAG response.

QUESTION: {query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Evaluate and respond ONLY in this exact JSON format (do not include markdown formatting or extra text outside the JSON block):
{{
  "context_relevance": <0.0 to 1.0>,
  "faithfulness": <0.0 to 1.0>,
  "answer_relevance": <0.0 to 1.0>,
  "answer_completeness": <0.0 to 1.0>,
  "overall_score": <0.0 to 1.0>,
  "failure_mode": "<none|bad_retrieval|hallucination|incomplete|off_topic>",
  "reasoning": "<one sentence explaining the score>"
}}

Scoring rules:
- context_relevance: what fraction of chunks are relevant to the question?
- faithfulness: are all claims in the answer supported by the context?
- answer_relevance: does the answer address what was actually asked?
- answer_completeness: does it cover all parts of the question?
- overall_score: weighted average (faithfulness weighted highest)
- failure_mode: the PRIMARY failure if overall_score < 0.75 (choose one from the list: none, bad_retrieval, hallucination, incomplete, off_topic)
"""


class AnswerEvaluator:
    """
    Evaluates RAG generation results using a hybrid heuristic + LLM-judge approach.
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        min_pass_score: float = 0.75,
        min_retrieval_similarity: float = 0.60,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.min_pass_score = min_pass_score
        self.min_retrieval_similarity = min_retrieval_similarity
        print(f"[Evaluator] Initialized — judge={model}, min_pass_score={min_pass_score}")

    def run_heuristics(self, query: str, search_results: list[dict], answer: str) -> list[str]:
        """
        Run fast, deterministic checks without calling an LLM.

        Args:
            query: Original user query.
            search_results: Chunks returned by FAISS search.
            answer: Generated LLM answer.

        Returns:
            List of detected flag strings.
        """
        flags = []

        # Check 1: Insufficient context admission
        insufficient_phrases = [
            "i don't have enough information",
            "the context does not contain",
            "i cannot find",
            "not mentioned in the provided",
            "not mentioned in the context",
            "insufficient information"
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in insufficient_phrases):
            flags.append("INSUFFICIENT_CONTEXT")

        # Check 2: Low retrieval scores
        if search_results:
            scores = [r["score"] for r in search_results]
            max_score = max(scores) if scores else 0.0
            if max_score < self.min_retrieval_similarity:
                flags.append("LOW_RETRIEVAL_SCORES")
        else:
            flags.append("LOW_RETRIEVAL_SCORES")

        # Check 3: Suspiciously short answer
        word_count = len(answer.split())
        if word_count < 10:
            flags.append("ANSWER_TOO_SHORT")

        # Check 4: Suspiciously long answer (verbose/padding)
        if word_count > 300:
            flags.append("ANSWER_TOO_LONG")

        # Check 5: Hedging language
        hedge_phrases = [
            "i think", "i believe", "probably",
            "i'm not sure", "might be", "could be", "it's possible"
        ]
        if any(phrase in answer_lower for phrase in hedge_phrases):
            flags.append("LOW_CONFIDENCE_LANGUAGE")

        return flags

    def compute_overall_score(self, scores: dict) -> float:
        """Calculate weighted score prioritizing faithfulness."""
        overall = sum(
            scores.get(dim, 0.0) * METRIC_WEIGHTS[dim]
            for dim in METRIC_WEIGHTS
        )
        return round(overall, 3)

    async def _query_llm_judge(self, query: str, context_str: str, answer: str) -> dict:
        """Call Groq to run the LLM-as-a-judge evaluation asynchronously."""
        formatted_prompt = EVAL_PROMPT.format(
            query=query,
            context=context_str,
            answer=answer
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.0,  # Deterministic for grading
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content.strip()
        
        try:
            # Clean up potential markdown formatting wrapping the JSON
            cleaned_json = re.sub(r"^```json\s*|\s*```$", "", raw_content, flags=re.MULTILINE)
            return json.loads(cleaned_json)
        except Exception as e:
            print(f"[Evaluator] Error parsing JSON from LLM Judge: {e}. Raw: {raw_content}")
            # Safe fallback response
            return {
                "context_relevance": 0.5,
                "faithfulness": 0.5,
                "answer_relevance": 0.5,
                "answer_completeness": 0.5,
                "overall_score": 0.5,
                "failure_mode": "hallucination",
                "reasoning": f"Failed to parse LLM Judge response: {e}"
            }

    async def evaluate(self, query: str, search_results: list[dict], formatted_context: str, answer: str) -> dict:
        """
        Orchestrate the hybrid evaluation flow asynchronously.

        Args:
            query: The user query.
            search_results: The raw search results list.
            formatted_context: The context block string fed to the generator.
            answer: The generated answer to evaluate.

        Returns:
            Dict containing:
              - "passed": bool
              - "overall_score": float
              - "failure_mode": str
              - "reasoning": str
              - "metrics": dict (the 4 dimensions)
              - "heuristic_flags": list[str]
        """
        print("[Evaluator] Running heuristic checks...")
        flags = self.run_heuristics(query, search_results, answer)
        print(f"[Evaluator] Heuristic flags: {flags}")

        # FAST PATH FAILURE: If we hit critical heuristic issues, fail immediately
        if "INSUFFICIENT_CONTEXT" in flags:
            return {
                "passed": False,
                "overall_score": 0.0,
                "failure_mode": "insufficient_context",
                "reasoning": "Heuristic check: Answer contains refusal phrases indicating insufficient context.",
                "metrics": {
                    "context_relevance": 0.0,
                    "faithfulness": 1.0,  # Factually faithful to nothing, but context is insufficient
                    "answer_relevance": 0.0,
                    "answer_completeness": 0.0
                },
                "heuristic_flags": flags
            }

        if "LOW_RETRIEVAL_SCORES" in flags:
            return {
                "passed": False,
                "overall_score": 0.0,
                "failure_mode": "bad_retrieval",
                "reasoning": "Heuristic check: Top retrieved chunk similarity score is below the minimum threshold.",
                "metrics": {
                    "context_relevance": 0.0,
                    "faithfulness": 1.0,
                    "answer_relevance": 0.0,
                    "answer_completeness": 0.0
                },
                "heuristic_flags": flags
            }

        # Otherwise, run LLM-as-a-judge for semantic evaluation
        print("[Evaluator] Running LLM-as-a-judge semantic evaluation...")
        judge_res = await self._query_llm_judge(query, formatted_context, answer)
        
        # Ensure overall score is re-calculated correctly according to weights
        metrics = {
            "context_relevance": float(judge_res.get("context_relevance", 0.0)),
            "faithfulness": float(judge_res.get("faithfulness", 0.0)),
            "answer_relevance": float(judge_res.get("answer_relevance", 0.0)),
            "answer_completeness": float(judge_res.get("answer_completeness", 0.0)),
        }
        
        overall_score = self.compute_overall_score(metrics)
        failure_mode = judge_res.get("failure_mode", "none").lower()
        reasoning = judge_res.get("reasoning", "")
        
        # Determine pass/fail based on overall score threshold
        passed = overall_score >= self.min_pass_score
        
        if passed:
            failure_mode = "none"
        elif failure_mode == "none":
            # If score is low but LLM didn't pick a failure mode, default to hallucination or bad_retrieval
            failure_mode = "hallucination" if metrics["faithfulness"] < metrics["context_relevance"] else "bad_retrieval"

        result = {
            "passed": passed,
            "overall_score": overall_score,
            "failure_mode": failure_mode,
            "reasoning": reasoning,
            "metrics": metrics,
            "heuristic_flags": flags
        }
        
        print(f"[Evaluator] Eval completed: Passed={passed}, Score={overall_score}, Mode={failure_mode}")
        return result
