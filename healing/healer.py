"""
healer.py — Query reformulation and healing module for the Self-Healing RAG pipeline.

Analyzes evaluation failures and generates targeted repair strategies:
  - bad_retrieval/insufficient_context: Reformulate query with alternative keywords/synonyms.
  - hallucination: Reformulate query to find direct support + generate strict grounding instructions.
  - incomplete: Increase retrieval parameter K + reformulate targeting missing details.
  - off_topic: Reformulate query to focus strictly on query intent.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

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

    def __init__(self, model: str = "llama3-70b-8192"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")

        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"[Healer] Initialized — model={model}")

    def _reformulate_query(self, query: str, failure_mode: str, reasoning: str, last_answer: str) -> str:
        """Call Groq to rewrite the query."""
        prompt = REFORMULATION_PROMPT.format(
            query=query,
            failure_mode=failure_mode,
            reasoning=reasoning,
            answer=last_answer
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Low temperature for focused keyword generation
            max_tokens=64
        )

        new_query = response.choices[0].message.content.strip()
        # Clean up any surrounding quotes if the LLM didn't follow formatting strictly
        new_query = new_query.strip('"').strip("'")
        return new_query

    def _generate_strict_directive(self, reasoning: str) -> str:
        """Generate a custom strict directive for hallucination avoidance."""
        prompt = STRICT_Grounding_PROMPT.format(reasoning=reasoning)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    def heal(self, original_query: str, last_answer: str, eval_result: dict, current_k: int) -> dict:
        """
        Produce a healing plan based on the evaluation result.

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

        # 1. Handle K parameter modification
        if failure_mode == "incomplete":
            # Retrieve more chunks to try and find missing answers
            new_k = current_k + 3
            print(f"[Healer] Increasing retrieval K from {current_k} to {new_k}")

        # 2. Handle Query Reformulation
        # We rewrite the query for retrieval failures, completeness, off-topic, insufficient context,
        # AND as requested by the user, we also do query reformulation for hallucinations.
        if failure_mode in ["bad_retrieval", "insufficient_context", "incomplete", "off_topic", "hallucination"]:
            try:
                healed_query = self._reformulate_query(original_query, failure_mode, reasoning, last_answer)
                print(f"[Healer] Reformulated query: '{original_query}' -> '{healed_query}'")
            except Exception as e:
                print(f"[Healer] Error reformulating query: {e}. Falling back to original query.")

        # 3. Add custom grounding directives for hallucinations
        if failure_mode == "hallucination":
            try:
                stricter_directive = self._generate_strict_directive(reasoning)
                print(f"[Healer] Generated anti-hallucination directive: {stricter_directive}")
            except Exception as e:
                print(f"[Healer] Error generating directive: {e}. Using standard grounding warning.")
                stricter_directive = "Ensure that EVERY claim you make is directly supported by the context. Do not add outside facts."

        return {
            "healed_query": healed_query,
            "stricter_directive": stricter_directive,
            "new_k": new_k
        }
