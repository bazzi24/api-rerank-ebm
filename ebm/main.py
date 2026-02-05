# import torch
# import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import List

# from ebm.model import JointEBMReranker

# # ================= CONFIG =================
# MODEL_PATH = "models/ebm_hardneg.pt"
# BASE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-v3"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # =========================================

# app = FastAPI(title="EBM Reranker (RAGFlow compatible)")

# # ===== Load model =====
# model = JointEBMReranker(
#     base_model_name=BASE_MODEL,
#     device=DEVICE,
#     freeze_encoder=True,   # inference dùng encoder
# )
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # -------- OpenAI-style schema --------
# class RerankRequest(BaseModel):
#     model: str | None = "ebm-reranker"
#     query: str
#     documents: List[str]
#     top_n: int | None = None


# class RerankResult(BaseModel):
#     index: int
#     relevance_score: float


# class RerankResponse(BaseModel):
#     results: List[RerankResult]


# # -------- Endpoint --------
# @app.post("/v1/rerank")
# def rerank(req: RerankRequest):
#     docs = req.documents
#     # print(req.documents)
#     query = req.query
#     top_n = req.top_n or len(docs)

#     if not docs:
#         return {"results": []}

#     with torch.no_grad():
#         queries = [query] * len(docs)

#         # [1, N]
#         energy_matrix = model.compute_energy_matrix(queries, docs)

#         # lấy hàng đầu tiên
#         energies = energy_matrix[0].cpu().numpy()

#     # energy thấp = tốt → đảo dấu
#     scores = -energies

#     # print("Energies:", energies)
#     # print("Scores:", scores)

#     # normalize về [0,1]
#     if scores.max() > scores.min():
#         scores = (scores - scores.min()) / (scores.max() - scores.min())
#     else:
#         scores = np.zeros_like(scores)

#     # sort giảm dần
#     idx = np.argsort(scores)[::-1][:top_n]

#     results = [
#         {
#             "index": int(i),
#             "relevance_score": float(scores[i])
            
#         }
#         for i in idx
#     ]
    
#     for i in idx:
#         print(f"Chunks: {docs}, Scores: {scores}")
    


#     return {"results": results}


##################
#
#   NEW VERSION 
#
#
##################


"""
Fixed EBM Reranker FastAPI - RAGFlow Compatible
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
import uuid
import os

# ================= CONFIG =================
MODEL_PATH = "models/ebm_hardneg.pt"
BASE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "log_ebm_api.log")
# =========================================

# Create a folder if it does not exists
os.makedirs(LOG_DIR, exist_ok=True)
#

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding="utf-8"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EBM Reranker (RAGFlow compatible)",
    version="1.0.0",
    description="Energy-Based Model Reranking API compatible with RAGFlow"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Request Logging Middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f">>> Incoming: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
        response = await call_next(request)
        logger.info(f"<<< Response: {response.status_code}")
        return response

app.add_middleware(RequestLoggingMiddleware)

# ===== Load model =====
logger.info(f"Loading EBM model from {MODEL_PATH}")
logger.info(f"Using device: {DEVICE}")

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
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise


# -------- Schemas (RAGFlow compatible) --------
class RerankRequest(BaseModel):
    """Request schema matching RAGFlow's expectations"""
    model: Optional[str] = Field(default="ebm-reranker", description="Model name (optional)")
    query: str = Field(..., description="Search query")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_n: Optional[int] = Field(default=None, description="Number of top results to return (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "documents": [
                    "Machine learning is a subset of AI",
                    "The weather is sunny today",
                    "Deep learning uses neural networks"
                ],
                "top_n": 3
            }
        }


class RerankResult(BaseModel):
    """Individual rerank result"""
    index: int = Field(..., description="Original document index")
    relevance_score: float = Field(..., description="Relevance score (0-1, higher is better)")
    document: str = Field(..., description="The original document text")


class RerankResponse(BaseModel):
    """Response schema matching RAGFlow's expectations"""
    id: str = Field(..., description="Unique request ID")
    results: List[RerankResult] = Field(..., description="Reranked results")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "results": [
                    {"index": 0, "relevance_score": 0.95, "document": "Machine learning is a subset of AI"},
                    {"index": 2, "relevance_score": 0.87, "document": "Deep learning uses neural networks"},
                    {"index": 1, "relevance_score": 0.12, "document": "The weather is sunny today"}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    version: str


# -------- Health Endpoint --------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=DEVICE,
        version="1.0.0"
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EBM Reranker API",
        "version": "1.0.0",
        "endpoints": {
            "rerank": "POST /v1/rerank",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


# -------- Rerank Endpoint (RAGFlow Compatible) --------
@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    """
    Rerank documents using EBM model.

    IMPORTANT: RAGFlow expects ALL documents to be scored and returned
    with their ORIGINAL indices, not just the top_n.

    Args:
        req: RerankRequest with query and documents

    Returns:
        RerankResponse with all documents scored and sorted by relevance
    """
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        docs = req.documents
        query = req.query
        top_n = req.top_n or len(docs)

        # Log request
        logger.info("=" * 80)
        logger.info(f"🔄 RERANK REQUEST at {start_time}")
        logger.info(f"   Query: {query[:100]}..." if len(query) > 100 else f"   Query: {query}")
        logger.info(f"   Documents: {len(docs)}")
        logger.info(f"   Top N requested: {top_n}")

        # Handle empty documents
        if not docs:
            logger.warning("⚠️  Empty documents list")
            return RerankResponse(results=[])

        # Compute scores with EBM model using batch processing to avoid GPU OOM
        BATCH_SIZE = 16  # Process 16 documents at a time to fit in 4GB GPU
        all_energies = []

        with torch.no_grad():
            num_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(docs))
                batch_docs = docs[start_idx:end_idx]

                # Process this batch
                queries = [query] * len(batch_docs)
                energy_matrix = model.compute_energy_matrix(queries, batch_docs)
                batch_energies = energy_matrix[0].cpu().numpy()
                all_energies.append(batch_energies)

                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine all batch results
            energies = np.concatenate(all_energies)

            logger.info(f"   Processed {num_batches} batches of size {BATCH_SIZE}")

        # Convert energy to scores (lower energy = better = higher score)
        scores = -energies

        # Normalize scores to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            # All scores are the same
            scores = np.zeros_like(scores)

        logger.info(f"   Energy range: [{energies.min():.4f}, {energies.max():.4f}]")
        logger.info(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")

        # Create results for ALL documents (not just top_n)
        # RAGFlow needs all documents scored with their original indices
        all_results = [
            RerankResult(
                index=i,
                relevance_score=float(scores[i]),
                document=docs[i]
            )
            for i in range(len(docs))
        ]

        # Sort by relevance score (descending)
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Return top_n if specified, otherwise all
        results_to_return = all_results[:top_n]

        # Log top results
        logger.info(f"   Top {min(5, len(results_to_return))} results:")
        for rank, result in enumerate(results_to_return[:5], 1):
            doc_preview = docs[result.index][:60] + "..." if len(docs[result.index]) > 60 else docs[result.index]
            logger.info(f"      {rank}. [idx={result.index}] score={result.relevance_score:.4f}: {doc_preview}")

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Reranking completed in {elapsed:.3f}s")
        logger.info(f"   Returning {len(results_to_return)} results")
        logger.info(f"   Request ID: {request_id}")
        logger.info("=" * 80)

        return RerankResponse(id=request_id, results=results_to_return)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"❌ RERANK ERROR: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Reranking failed: {str(e)}"
        )


# -------- Startup Event --------
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("=" * 80)
    logger.info("🚀 EBM Reranker API Started")
    logger.info("=" * 80)
    logger.info(f"Model: {BASE_MODEL}")
    logger.info(f"Weights: {MODEL_PATH}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Endpoints:")
    logger.info(f"  - POST /v1/rerank")
    logger.info(f"  - GET /health")
    logger.info(f"  - GET /docs")
    logger.info("=" * 80)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )




