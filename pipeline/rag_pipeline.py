"""
rag_pipeline.py — Main orchestrator for the Self-Healing RAG pipeline.

Integrates ingestion (loader, chunker), retrieval (embedder, vector store),
generation (generator), evaluation (evaluator), and healing (healer) into
a closed-loop feedback pipeline with detailed execution tracing.
"""

import os
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from retrieval.embedder import Embedder
from retrieval.vector_store import VectorStore
from generation.generator import Generator, SYSTEM_PROMPT
from evaluation.evaluator import AnswerEvaluator
from healing.healer import QueryHealer


class SelfHealingRAGPipeline:
    """
    Orchestrates the entire self-healing RAG loop.

    Flow:
      1. Retrieve context using query.
      2. Generate answer with context.
      3. Evaluate answer (heuristics + LLM-as-a-judge).
      4. If evaluation passes, return answer.
      5. If evaluation fails and retries remain:
         a. Invoke healer to reformulate query, update retrieval K,
            and/or generate strict LLM directives.
         b. Loop back to step 1 with updated parameters.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        llm_model: str = "llama3-70b-8192",
        default_k: int = 5,
        max_retries: int = 3,
        min_pass_score: float = 0.75,
        min_retrieval_similarity: float = 0.60,
    ):
        """
        Initialize all RAG pipeline sub-components.
        """
        self.default_k = default_k
        self.max_retries = max_retries

        # Ingestion & Retrieval Components
        self.embedder = Embedder(model_name=model_name)
        self.vector_store = VectorStore(dimension=self.embedder.dimension, top_k=default_k)

        # LLM Generation, Evaluation, and Healing Components
        self.generator = Generator(model=llm_model)
        self.evaluator = AnswerEvaluator(
            model=llm_model,
            min_pass_score=min_pass_score,
            min_retrieval_similarity=min_retrieval_similarity
        )
        self.healer = QueryHealer(model=llm_model)

        print("[Pipeline] Self-Healing RAG Pipeline successfully initialized.")

    def ingest_directory(self, data_dir: str = "data", chunk_size: int = 512, chunk_overlap: int = 100) -> int:
        """
        Run the complete ingestion flow for a directory:
        Load -> Chunk -> Embed -> Index in FAISS.

        Args:
            data_dir: Path to directory containing documents.
            chunk_size: Text chunking size.
            chunk_overlap: Overlap between chunks.

        Returns:
            Number of chunks indexed.
        """
        print(f"[Pipeline] Starting ingestion from directory: {data_dir}...")
        self.vector_store.reset()

        # 1. Load documents
        documents = load_documents(data_dir)
        if not documents:
            print("[Pipeline] No documents found to ingest.")
            return 0

        # 2. Chunk documents
        chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            print("[Pipeline] No chunks created.")
            return 0

        # 3. Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)

        # 4. Add to Vector Store
        self.vector_store.add(embeddings, chunks)
        print(f"[Pipeline] Ingestion complete. Indexed {len(chunks)} chunks.")
        return len(chunks)

    def query(self, user_query: str) -> dict:
        """
        Execute a query against the self-healing RAG pipeline.

        Loops through Retrieval -> Generation -> Evaluation -> Healing
        until the answer passes evaluation or max_retries is reached.

        Args:
            user_query: The question asked by the user.

        Returns:
            Dict containing:
              - "query": Original query (str).
              - "answer": Final generated answer (str).
              - "passed": Whether the final evaluation passed (bool).
              - "retries_attempted": Number of healing retries done (int).
              - "trace": Full step-by-step history of attempts (list[dict]).
        """
        trace = []
        current_query = user_query
        current_k = self.default_k
        stricter_directive = None
        attempt = 0

        print(f"\n[Pipeline] Processing new query: '{user_query}'")

        while attempt <= self.max_retries:
            print(f"\n[Pipeline] --- Attempt {attempt} (Max Retries: {self.max_retries}) ---")
            print(f"[Pipeline] Retrieval Query: '{current_query}' | K: {current_k}")

            # 1. Embed current query
            query_vector = self.embedder.embed_query(current_query)

            # 2. Retrieve top-K chunks from FAISS
            search_results = self.vector_store.search(query_vector, top_k=current_k)
            print(f"[Pipeline] Retrieved {len(search_results)} chunk(s).")

            # 3. Generate answer
            # If we have a stricter directive from a previous healing loop, temporarily apply it.
            original_generator_model = self.generator.model
            if stricter_directive:
                print(f"[Pipeline] Applying stricter directive: '{stricter_directive}'")
                # Temporarily replace system prompt in Generator
                from generation import generator as gen_mod
                old_prompt = gen_mod.SYSTEM_PROMPT
                gen_mod.SYSTEM_PROMPT = f"{old_prompt}\n\nCRITICAL DIRECTIVE:\n{stricter_directive}"

            try:
                gen_res = self.generator.generate(user_query, search_results)
                generated_answer = gen_res["answer"]
                context_used = gen_res["context_used"]
            finally:
                # Always restore system prompt
                if stricter_directive:
                    gen_mod.SYSTEM_PROMPT = old_prompt

            # 4. Evaluate generated answer
            eval_res = self.evaluator.evaluate(user_query, search_results, context_used, generated_answer)

            # Record this attempt in trace history
            trace_item = {
                "attempt": attempt,
                "query_used": current_query,
                "retrieval_k": current_k,
                "retrieved_chunks": [
                    {
                        "source": r["metadata"].get("source", "unknown"),
                        "page": r["metadata"].get("page", "?"),
                        "score": r["score"],
                        "content": r["content"]
                    }
                    for r in search_results
                ],
                "generated_answer": generated_answer,
                "evaluation": eval_res
            }
            trace.append(trace_item)

            # Check if evaluation passed
            if eval_res["passed"]:
                print(f"[Pipeline] PASS: Evaluation passed on attempt {attempt}!")
                return {
                    "query": user_query,
                    "answer": generated_answer,
                    "passed": True,
                    "retries_attempted": attempt,
                    "trace": trace
                }

            # If evaluation failed, prepare healing for next loop
            if attempt == self.max_retries:
                print("[Pipeline] FAIL: Max retries reached without passing evaluation.")
                break

            print(f"[Pipeline] FAIL: Evaluation failed. Triggering healer...")
            healing_plan = self.healer.heal(
                original_query=user_query,
                last_answer=generated_answer,
                eval_result=eval_res,
                current_k=current_k
            )

            # Update parameters for the next iteration
            current_query = healing_plan["healed_query"]
            stricter_directive = healing_plan["stricter_directive"]
            current_k = healing_plan["new_k"]
            attempt += 1

        # Return the final attempt's results if healing failed to pass within max_retries
        return {
            "query": user_query,
            "answer": trace[-1]["generated_answer"],
            "passed": False,
            "retries_attempted": self.max_retries,
            "trace": trace
        }
