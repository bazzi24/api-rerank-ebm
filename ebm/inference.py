import torch
from ebm.model import JointEBMReranker

device = "cuda" if torch.cuda.is_available() else "cpu"

model = JointEBMReranker(
    base_model_name="sentence-transformers/msmarco-MiniLM-L6-v3",
    device=device,
)

model.load_state_dict(
    torch.load("models/ebm_reranker_final.pt", map_location=device)
)
model.to(device)
model.eval()
