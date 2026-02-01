import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm

from model import JointEBMReranker
from utils import ebm_inbatch_softmax_loss, collate_fn

with open("../config.yaml") as f:
    config = yaml.safe_load(f)

accelerator = Accelerator()

dataset = load_dataset(
    config["dataset"]["name"],
    config["dataset"]["subset"],
    split=config["dataset"]["split"],
    cache_dir=config["dataset"]["cache_dir"],
)

train_loader = DataLoader(
    dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
)

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

model.train()
for epoch in range(config["training"]["epochs"]):
    total = 0.0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for queries, positives in progress:
        if len(queries) < 2:
            continue

        energies = model.compute_energy_matrix(queries, positives)
        loss = ebm_inbatch_softmax_loss(energies)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    print(f"Epoch {epoch+1} | Avg Loss: {total / len(train_loader):.4f}")

unwrapped = accelerator.unwrap_model(model)
torch.save(
    unwrapped.state_dict(),
    f"{config['training']['save_dir']}/ebm_reranker_final.pt",
)

print("Training completed")
