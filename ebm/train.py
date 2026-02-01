import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm

from model import JointEBMReranker
from utils import ebm_inbatch_loss, collate_fn

# ===== Load config =====
with open("../config.yaml") as f:
    config = yaml.safe_load(f)

accelerator = Accelerator()

# ===== Dataset =====
dataset = load_dataset("microsoft/ms_marco", "v2.1")["train"]

train_loader = DataLoader(
    dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

# ===== Model =====
model = JointEBMReranker(
    base_model_name=config["model"]["base_model"],
    freeze_encoder=config["training"]["freeze_encoder"],
)

optimizer = AdamW(
    model.parameters(),
    lr=float(config["training"]["learning_rate"]),
)

model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

# ===== Train =====
model.train()
for epoch in range(config["training"]["epochs"]):
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for queries, positives in progress:
        energies = model(queries, positives)

        loss = ebm_inbatch_loss(
            energies,
            margin=config["training"]["margin"],
        )

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg:.4f}")

# ===== Save =====
unwrapped = accelerator.unwrap_model(model)
torch.save(
    unwrapped.state_dict(),
    f"{config['training']['save_dir']}/ebm_reranker_final.pt",
)

print("Training completed")
