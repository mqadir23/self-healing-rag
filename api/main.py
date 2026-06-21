"""
main.py — FastAPI API layer for the Self-Healing RAG pipeline.

Exposes endpoints for:
  - Document ingestion (/ingest): scan data/ and upload files.
  - Query processing (/query): run query through self-healing loop.
  - Index management (/status, /reset): inspect and clear index.
  - Health check (/health).
"""

import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pipeline.rag_pipeline import SelfHealingRAGPipeline

# Global pipeline instance initialized on startup
pipeline = None
DATA_DIR = "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize models and directories on startup."""
    global pipeline
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    print("[API] Initializing Self-Healing RAG Pipeline (loading models)...")
    pipeline = SelfHealingRAGPipeline()
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Self-Healing RAG Pipeline API",
    description="A production-grade RAG pipeline featuring automated evaluation and query-healing loops.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend/testing integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask the RAG pipeline.")


class QueryResponse(BaseModel):
    query: str
    answer: str
    passed: bool
    retries_attempted: int
    trace: list


@app.get("/health", tags=["Diagnostic"])
async def health_check():
    """Verify API is alive and models are loaded."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is initializing or unavailable.")
    return {"status": "healthy", "pipeline_initialized": True}


@app.get("/status", tags=["Diagnostic"])
async def index_status():
    """Check how many document chunks are currently indexed in FAISS."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is initializing.")

    num_vectors = pipeline.vector_store.index.ntotal
    unique_docs = list(set(chunk.metadata.get("source", "unknown") for chunk in pipeline.vector_store.chunks))

    return {
        "indexed_chunks": num_vectors,
        "indexed_documents": unique_docs,
        "num_documents": len(unique_docs)
    }


@app.post("/reset", tags=["Diagnostic"])
async def reset_index():
    """Clear all indexed documents and embeddings."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is initializing.")

    await pipeline.vector_store.reset()
    return {"message": "FAISS index and metadata store successfully cleared."}


@app.post("/ingest", tags=["Ingestion"])
async def ingest_documents(
    files: list[UploadFile | str] | None = File(None, description="Optional files to upload and ingest. If omitted, scans data/ folder.")
):
    """
    Ingest documents into the RAG vector store.

    If files are uploaded, they are saved to the `data/` directory.
    Then, the API scans the `data/` directory and rebuilds the FAISS index.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is initializing.")

    # 1. Save uploaded files to data/ if present
    valid_files = []
    if files:
        for f in files:
            if isinstance(f, str):
                continue
            if getattr(f, "filename", "") == "":
                continue
            valid_files.append(f)

    if valid_files:
        for file in valid_files:
            file_path = os.path.join(DATA_DIR, file.filename)
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                print(f"[API] Uploaded and saved file: {file.filename}")
            except Exception as e:
                print(f"[API] Failed to save uploaded file {file.filename}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to save file {file.filename}: {e}")

    # 2. Run ingestion pipeline
    try:
        num_chunks = await pipeline.ingest_directory(DATA_DIR)
        return {
            "message": "Ingestion completed successfully.",
            "total_chunks_indexed": num_chunks,
            "data_directory": os.path.abspath(DATA_DIR)
        }
    except Exception as e:
        print(f"[API] Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
async def query_pipeline(request: QueryRequest):
    """
    Query the self-healing RAG pipeline.

    Evaluates the response and attempts query healing/re-retrieval
    up to 3 times before returning the final trace and answer.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is initializing.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = await pipeline.query(request.query)
        return result
    except Exception as e:
        print(f"[API] Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")
