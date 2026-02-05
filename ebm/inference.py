import torch
from ebm.models.energy_head import EBMEnergyHead
from ebm.models.text_reranker import TextEBMReranker

device = "cuda" if torch.cuda.is_available() else "cpu"

energy = EBMEnergyHead(hidden_size=384)
energy.load_state_dict(torch.load("models/energy_head.pt"))
energy.eval()

model = TextEBMReranker(
    base_model="sentence-transformers/msmarco-MiniLM-L6-v3",
    energy_head=energy,
    device=device,
)

docs, scores = model.rerank(
    "What is the capital of France?",
    ["Paris is the capital of France.", "Berlin is in Germany."],
)
print(docs, scores)
