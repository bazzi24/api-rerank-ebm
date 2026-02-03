# ebm/main.py
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from ebm.model import JointEBMReranker

# ================= CONFIG =================
MODEL_PATH = "models/ebm_hardneg.pt"
BASE_MODEL = "sentence-transformers/msmarco-MiniLM-L6-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

app = FastAPI(title="EBM Reranker (RAGFlow compatible)")

# ===== Load model =====
model = JointEBMReranker(
    base_model_name=BASE_MODEL,
    device=DEVICE,
    freeze_encoder=True,   # inference dùng encoder
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------- OpenAI-style schema --------
class RerankRequest(BaseModel):
    model: str | None = "ebm-reranker"
    query: str
    documents: List[str]
    top_n: int | None = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: List[RerankResult]


# -------- Endpoint --------
@app.post("/v1/rerank")
def rerank(req: RerankRequest):
    docs = req.documents
    query = req.query
    top_n = req.top_n or len(docs)

    if not docs:
        return {"results": []}

    with torch.no_grad():
        queries = [query] * len(docs)

        # [1, N]
        energy_matrix = model.compute_energy_matrix(queries, docs)

        # lấy hàng đầu tiên
        energies = energy_matrix[0].cpu().numpy()

    # energy thấp = tốt → đảo dấu
    scores = -energies

    print("Energies:", energies)
    print("Scores:", scores)

    # normalize về [0,1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.zeros_like(scores)

    # sort giảm dần
    idx = np.argsort(scores)[::-1][:top_n]

    results = [
        {
            "index": int(i),
            "relevance_score": float(scores[i])
        }
        for i in idx
    ]

    return {"results": results}




