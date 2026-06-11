"""
generator.py — Groq LLM generation for the Self-Healing RAG pipeline.

Uses llama3-70b-8192 via the Groq API with a structured RAG prompt
that enforces context-only answering with citations.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


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

        self.client = Groq(api_key=api_key)
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

    def generate(self, query: str, search_results: list[dict]) -> dict:
        """
        Generate an answer for the given query using retrieved context.

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content.strip()

        print(f"[Generator] Answer generated — {len(answer)} chars")

        return {
            "answer": answer,
            "model": self.model,
            "context_used": context_block,
        }
