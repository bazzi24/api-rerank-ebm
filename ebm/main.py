from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch

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