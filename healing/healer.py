"""
healer.py — Query reformulation and healing module for the Self-Healing RAG pipeline.

Analyzes evaluation failures and generates targeted repair strategies:
  - bad_retrieval/insufficient_context: Reformulate query with alternative keywords/synonyms.
  - hallucination: Reformulate query to find direct support + generate strict grounding instructions.
  - incomplete: Increase retrieval parameter K + reformulate targeting missing details.
  - off_topic: Reformulate query to focus strictly on query intent.
"""

import os
import logging
import groq as groq_module
from groq import AsyncGroq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

load_dotenv()

logger = logging.getLogger(__name__)

# Groq transient errors safe to retry
_GROQ_RETRYABLE = (
    groq_module.RateLimitError,
    groq_module.APIConnectionError,
    groq_module.APITimeoutError,
    groq_module.InternalServerError,
)
REFORMULATION_PROMPT = """You are a search query reformulation assistant.
An evaluation of a Retrieval-Augmented Generation (RAG) pipeline response failed.

Original User Query: {query}
Failure Mode: {failure_mode}
Evaluator Feedback: {reasoning}
Last Generated Answer: {answer}

Your job is to write a single, optimized search query to retrieve better context from a FAISS vector store.
Guidelines:
1. Focus on keywords, synonyms, and key entities. Do not ask conversational questions.
2. If the failure was "incomplete", focus the new query on the missing parts of the question.
3. If the failure was "hallucination", target the core fact that was hallucinated.
4. If the failure was "bad_retrieval" or "insufficient_context", expand with broader synonyms or related concepts.

Respond ONLY with the new query string. Do not use quotes, markdown formatting, or introductory phrases.
New Query:"""

STRICT_Grounding_PROMPT = """You are a query healing assistant.
The last response hallucinated or added information not present in the context.
Evaluator Feedback: {reasoning}

Create a single strict system instruction/directive that we can pass to the generator to prevent it from repeating this hallucination. Keep it under 2 sentences.
Instruction:"""


class QueryHealer:
    """
    Formulates healing strategies based on evaluation failures.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        print(f"[Healer] Initialized — model={model}")

    @retry(
        retry=retry_if_exception_type(_GROQ_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_groq(self, messages: list[dict]) -> str:
        """Unified Groq API call with retry logic."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=64,
        )
        return response.choices[0].message.content.strip()

    async def _reformulate_query(self, query: str, failure_mode: str, reasoning: str, last_answer: str) -> str:
        """Call Groq to rewrite the query asynchronously."""
        prompt = REFORMULATION_PROMPT.format(
            query=query,
            failure_mode=failure_mode,
            reasoning=reasoning,
            answer=last_answer
        )
        messages = [{"role": "user", "content": prompt}]
        new_query = await self._call_groq(messages)
        # Clean up any surrounding quotes if the LLM didn't follow formatting strictly
        new_query = new_query.strip('"').strip("'")
        return new_query

    @retry(
        retry=retry_if_exception_type(_GROQ_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _generate_strict_directive(self, reasoning: str) -> str:
        """Generate a custom strict directive for hallucination avoidance with retry."""
        prompt = STRICT_Grounding_PROMPT.format(reasoning=reasoning)
        messages = [{"role": "user", "content": prompt}]
        return await self._call_groq(messages)

    async def heal(self, original_query: str, last_answer: str, eval_result: dict, current_k: int) -> dict:
        """
        Produce a healing plan based on the evaluation result asynchronously.

        Args:
            original_query: The original question asked by the user.
            last_answer: The answer that failed evaluation.
            eval_result: The evaluation results dict containing 'failure_mode' and 'reasoning'.
            current_k: The number of chunks currently retrieved.

        Returns:
            Dict containing:
              - "healed_query": Reformulated search query (str).
              - "stricter_directive": Extra instructions for the LLM generator (str or None).
              - "new_k": Updated number of chunks to retrieve (int).
        """
        failure_mode = eval_result.get("failure_mode", "none").lower()
        reasoning = eval_result.get("reasoning", "")

        print(f"[Healer] Formulating healing strategy for failure_mode: {failure_mode}")

        healed_query = original_query
        stricter_directive = None
        new_k = current_k
        # Default balanced weights
        dense_weight = 0.5
        sparse_weight = 0.5

        # 1. Handle K parameter modification
        if failure_mode == "incomplete":
            # Retrieve more chunks to try and find missing answers
            new_k = current_k + 3
            print(f"[Healer] Increasing retrieval K from {current_k} to {new_k}")

        # 2. Adjust retrieval weights based on failure mode
        if failure_mode in ("bad_retrieval", "insufficient_context"):
            # Boost sparse (BM25) to capture exact keyword matches
            dense_weight = 0.3
            sparse_weight = 0.7
            print(f"[Healer] Boosting sparse weight to {sparse_weight} for keyword-focused retrieval.")
        elif failure_mode == "hallucination":
            # Boost dense to anchor retrieval on semantic context
            dense_weight = 0.7
            sparse_weight = 0.3
            print(f"[Healer] Boosting dense weight to {dense_weight} for semantic-focused retrieval.")

        # 3. Handle Query Reformulation
        # We rewrite the query for retrieval failures, completeness, off-topic, insufficient context,
        # AND as requested by the user, we also do query reformulation for hallucinations.
        if failure_mode in ["bad_retrieval", "insufficient_context", "incomplete", "off_topic", "hallucination"]:
            try:
                healed_query = await self._reformulate_query(original_query, failure_mode, reasoning, last_answer)
                print(f"[Healer] Reformulated query: '{original_query}' -> '{healed_query}'")
            except Exception as e:
                print(f"[Healer] Error reformulating query: {e}. Falling back to original query.")

        # 4. Add custom grounding directives for hallucinations
        if failure_mode == "hallucination":
            try:
                stricter_directive = await self._generate_strict_directive(reasoning)
                print(f"[Healer] Generated anti-hallucination directive: {stricter_directive}")
            except Exception as e:
                print(f"[Healer] Error generating directive: {e}. Using standard grounding warning.")
                stricter_directive = "Ensure that EVERY claim you make is directly supported by the context. Do not add outside facts."

        return {
            "healed_query": healed_query,
            "stricter_directive": stricter_directive,
            "new_k": new_k,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }
