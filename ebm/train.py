# test
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from accelerate import Accelerator
from tqdm import tqdm

from model import JointEBMReranker
from utils import ebm_hardneg_loss

# ================= CONFIG =================
CACHE_PATH = "cache/hardneg_cache.pt"
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda"
# =========================================


def main():
    accelerator = Accelerator()
    device = accelerator.device

    # ===== Load cache =====
    data = torch.load(CACHE_PATH, map_location="cpu")
    q = data["query"]
    p = data["positive"]
    neg = data["negatives"]  # [N, K, D]

    dataset = TensorDataset(q, p, neg)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ===== Model (ENERGY HEAD ONLY) =====
    model = JointEBMReranker(
        base_model_name="sentence-transformers/msmarco-MiniLM-L6-v3",
        freeze_encoder=True, 
        device=DEVICE,
    )

    optimizer = AdamW(model.energy_head.parameters(), lr=LR)

    model, optimizer, loader = accelerator.prepare(
        model, optimizer, loader
    )

    # ===== Train =====
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        prog = tqdm(loader, desc=f"Epoch {epoch+1}")
        for q_emb, p_emb, neg_emb in prog:
            # q_emb, p_emb: [B, D]
            # neg_emb: [B, K, D]

            pos_energy = model.energy_head(p_emb).squeeze(-1)   # [B]
            neg_energy = model.energy_head(neg_emb).squeeze(-1) # [B, K]

            loss = ebm_hardneg_loss(pos_energy, neg_energy)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} | Avg Loss: {total_loss / len(loader):.4f}")

    # ===== Save =====
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), "models/ebm_hardneg.pt")
    print(" Training finished & model saved")


if __name__ == "__main__":
    main()
