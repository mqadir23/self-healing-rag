"""
generator.py — Groq LLM generation for the Self-Healing RAG pipeline.

Uses llama3-70b-8192 via the Groq API with a structured RAG prompt
that enforces context-only answering with citations.

Reliability: Groq API calls are wrapped with tenacity exponential-backoff retry
(up to 4 attempts, 2–30s wait) to handle transient RateLimitError,
APIConnectionError, APITimeoutError, and InternalServerError (5xx) failures.
"""

import os
import logging
import groq as groq_module
from groq import AsyncGroq
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Groq transient errors that are safe to retry
_GROQ_RETRYABLE = (
    groq_module.RateLimitError,
    groq_module.APIConnectionError,
    groq_module.APITimeoutError,
    groq_module.InternalServerError,
)


SYSTEM_PROMPT = """You are a precise question-answering assistant.

Rules:
1. Answer ONLY using the provided context below.
2. If the context does not contain enough information to answer the question, say exactly: "I don't have enough information to answer this."
3. Do not use your training knowledge.
4. Do not make assumptions beyond what is stated.
5. Be concise and direct.
6. If you use information from the context, reference which part supports your answer."""


class Generator:
    """
    Generates answers using Groq's LLM API with structured RAG prompts.

    The prompt structure:
      - System: Strict rules for context-only answering with citations.
      - User: Structured context block (source, page, relevance) + question
              with a re-anchoring line before generation.

    Reliability:
      - Retries up to 4 times on transient Groq errors (rate limits, timeouts,
        connection drops, 5xx) with exponential backoff (2s → 30s, with jitter).

    Attributes:
        model: The Groq model identifier.
        temperature: Sampling temperature (0.3 — balanced factuality).
        max_tokens: Maximum tokens in the generated answer.
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        print(f"[Generator] Initialized — model={model}, temp={temperature}")

    def _format_context(self, search_results: list[dict]) -> str:
        """
        Format retrieved search results into a structured context block.

        Each chunk is tagged with source, page, and relevance score so the
        LLM can cite and weigh information appropriately.

        Args:
            search_results: List of dicts from VectorStore.search(), each with
                            'content', 'metadata', and 'score' keys.

        Returns:
            Formatted context string.
        """
        context_parts = []
        separator = "─" * 50

        for result in search_results:
            meta = result.get("metadata", {})
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            score = result.get("score", 0.0)

            context_parts.append(
                f"[Source: {source} | Page: {page} | Relevance: {score:.2f}]\n"
                f"{result['content']}"
            )

        formatted = f"\nCONTEXT:\n{separator}\n"
        formatted += f"\n\n".join(context_parts)
        formatted += f"\n{separator}"

        return formatted

    @retry(
        retry=retry_if_exception_type(_GROQ_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_groq(self, user_message: str) -> str:
        """
        Inner API call wrapped with retry logic.

        Separated from generate() so retries only hit the network call,
        not context formatting or result assembly.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    async def generate(self, query: str, search_results: list[dict]) -> dict:
        """
        Generate an answer for the given query using retrieved context asynchronously.

        Args:
            query: The user's question.
            search_results: Retrieved chunks from VectorStore.search().

        Returns:
            Dict with:
              - "answer": The generated answer text.
              - "model": Model used.
              - "context_used": The formatted context sent to the LLM.
        """
        context_block = self._format_context(search_results)

        user_message = (
            f"{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            f"Answer based strictly on the context above:"
        )

        # messages list not needed; _call_groq builds messages internally




        answer = await self._call_groq(user_message)

        print(f"[Generator] Answer generated — {len(answer)} chars")

        return {
            "answer": answer,
            "model": self.model,
            "context_used": context_block,
        }
