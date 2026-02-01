from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import numpy as np

from ebm.model import JointEBMReranker

app = FastAPI(title="EBM Reranker API")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = JointEBMReranker(
    base_model_name="sentence-transformers/msmarco-MiniLM-L6-v3",
    device=device,
)
model.load_state_dict(torch.load("models/ebm_reranker_final.pt", map_location=device))
model.to(device)
model.eval()


# ----------- SCHEMA (OpenAI-style) ----------------
class RerankRequest(BaseModel):
    model: str = "ebm-reranker"
    query: str
    documents: List[str]
    top_n: int | None = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


# ----------- ENDPOINT ----------------
@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    docs = req.documents
    query = req.query
    top_n = req.top_n or len(docs)

    if not docs:
        return {"results": []}

    with torch.no_grad():
        queries = [query] * len(docs)
        energies = model.compute_energy_batch(queries, docs)
        energies = energies.cpu().numpy()

    # energy thấp = tốt → score cao
    scores = -energies

    # normalize [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    indices = np.argsort(scores)[::-1][:top_n]

    results = [
        {"index": int(i), "relevance_score": float(scores[i])}
        for i in indices
    ]

    return {"results": results}
