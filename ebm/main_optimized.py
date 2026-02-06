"""
Optimized EBM Reranker FastAPI - Simplified for RAGFlow
Based on actual RAGFlow BazziReRank requirements
"""

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from datetime import datetime

# ================= CONFIG =================
MODEL_PATH = "models/ebm_hardneg.pt"
BASE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EBM Reranker (RAGFlow Compatible)",
    version="2.0.0",
    description="Optimized Energy-Based Model Reranking API for RAGFlow"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        logger.info(f">>> {request.method} {request.url.path} from {client_ip}")
        response = await call_next(request)
        logger.info(f"<<< {response.status_code}")
        return response

app.add_middleware(RequestLoggingMiddleware)

# ===== Load model =====
logger.info(f"Loading EBM model from {MODEL_PATH}")
logger.info(f"Device: {DEVICE}")

try:
    from ebm.model import JointEBMReranker

    model = JointEBMReranker(
        base_model_name=BASE_MODEL,
        device=DEVICE,
        freeze_encoder=True,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    logger.info(" Model loaded successfully")
except Exception as e:
    logger.error(f" Model loading failed: {e}")
    raise


# ======== Schemas ========
class RerankRequest(BaseModel):
    """Simple request schema - RAGFlow compatible"""
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="Documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of results (optional)")
    model: Optional[str] = Field(default=None, description="Model name (optional, ignored)")


class RerankResult(BaseModel):
    """Simple result schema - what RAGFlow actually uses"""
    index: int = Field(..., description="Original document index")
    relevance_score: float = Field(..., description="Relevance score (0-1)")


class RerankResponse(BaseModel):
    """Simple response schema - RAGFlow compatible"""
    results: List[RerankResult] = Field(..., description="Reranked results")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    version: str


# ======== Endpoints ========
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EBM Reranker API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "rerank": "POST /v1/rerank",
            "health": "GET /health",
            "docs": "GET /docs",
            "test": "POST /test-rerank"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=DEVICE,
        version="2.0.0"
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """
    Rerank documents using EBM model.

    This is the main endpoint called by RAGFlow's BazziReRank class.
    Returns ALL documents with scores, sorted by relevance.
    """
    start_time = datetime.now()

    try:
        docs = req.documents
        query = req.query
        top_n = req.top_n or len(docs)

        # Log request
        logger.info("=" * 80)
        logger.info(f"   RERANK REQUEST at {start_time}")
        logger.info(f"   Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
        logger.info(f"   Documents: {len(docs)}")
        logger.info(f"   Top N: {top_n}")

        # Log first few documents for debugging
        for i, doc in enumerate(docs[:3]):
            preview = doc[:60] + "..." if len(doc) > 60 else doc
            logger.info(f"   Doc[{i}]: {preview}")
        if len(docs) > 3:
            logger.info(f"   ... and {len(docs) - 3} more documents")

        # Handle empty documents
        if not docs:
            logger.warning("  Empty documents list")
            return RerankResponse(results=[])

        # Compute EBM scores
        with torch.no_grad():
            queries = [query] * len(docs)
            energy_matrix = model.compute_energy_matrix(queries, docs)
            energies = energy_matrix[0].cpu().numpy()

        # Convert energy to scores (lower energy = better)
        scores = -energies

        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            scores = np.zeros_like(scores)

        logger.info(f"   Energy: [{energies.min():.4f}, {energies.max():.4f}]")
        logger.info(f"   Scores: [{scores.min():.4f}, {scores.max():.4f}]")

        # Create results for ALL documents
        all_results = [
            RerankResult(index=i, relevance_score=float(scores[i]))
            for i in range(len(docs))
        ]

        # Sort by score (descending)
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Return top_n
        results = all_results[:top_n]

        # Log top results
        logger.info(f"   Top {min(3, len(results))} results:")
        for rank, r in enumerate(results[:3], 1):
            doc_preview = docs[r.index][:50] + "..." if len(docs[r.index]) > 50 else docs[r.index]
            logger.info(f"      {rank}. [idx={r.index}] score={r.relevance_score:.4f} - {doc_preview}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f" Completed in {elapsed:.3f}s - Returning {len(results)} results")
        logger.info("=" * 80)

        return RerankResponse(results=results)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f" RERANK ERROR: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-rerank")
async def test_rerank():
    """
    Test endpoint to verify rerank is working.
    Doesn't require RAGFlow - useful for debugging.
    """
    test_query = "What is machine learning?"
    test_docs = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "The weather today is sunny with a high of 75 degrees.",
        "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
        "Python is a popular programming language used in data science.",
        "Java is used for enterprise application development."
    ]

    logger.info(" Running test rerank...")

    result = await rerank(RerankRequest(
        query=test_query,
        documents=test_docs,
        top_n=5
    ))

    # Format response for easy reading
    return {
        "test_query": test_query,
        "results": [
            {
                "rank": i + 1,
                "index": r.index,
                "score": round(r.relevance_score, 4),
                "document": test_docs[r.index]
            }
            for i, r in enumerate(result.results)
        ],
        "status": " Test passed!"
    }


# ======== Startup ========
@app.on_event("startup")
async def startup():
    """Log startup info"""
    logger.info("=" * 80)
    logger.info(" EBM Reranker API Started")
    logger.info("=" * 80)
    logger.info(f"Version: 2.0.0 (Optimized)")
    logger.info(f"Model: {BASE_MODEL}")
    logger.info(f"Weights: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info("Endpoints:")
    logger.info("  POST /v1/rerank - Main rerank endpoint")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /test-rerank - Test endpoint")
    logger.info("  GET  /docs - API documentation")
    logger.info("=" * 80)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
