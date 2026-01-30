from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from ebm.model import JointEBMReranker
from ebm.inference import model, rerank

import numpy as np

app = FastAPI(title="EBM Reranker")

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
    
@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    docs = req.documents
    query = req.query
    top_n = req.top_n or len(docs)
    
    if not docs:
        return {"results": []}
    
    with torch.no_grad():
        energies = model.rerank(query, docs)
        energies = energies.numpy()
    
    # Low energies = good -> high score    
    scores = -energies
    
    # Normalize [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    indices = np.argsort(scores)[::-1][:top_n]
    
    results = [
        {"index": int(i), "relevance_score": float(scores[i])}
        for i in indices
    ]
    
    return {"results": results}