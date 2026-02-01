# ebm/train.py
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

from ebm.models.energy_head import EBMEnergyHead
from ebm.utils import ebm_hardneg_loss

accelerator = Accelerator()
device = accelerator.device

data = torch.load("cache/hardneg_cache.pt")
dataset = TensorDataset(data["q"], data["p"], data["neg"])
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = EBMEnergyHead(hidden_size=data["q"].size(1))
optimizer = AdamW(model.parameters(), lr=1e-4)

model, optimizer, loader = accelerator.prepare(
    model, optimizer, loader
)

model.train()
for epoch in range(5):
    total = 0
    for q, p, neg in loader:
        pos_e = model(q + p)
        neg_e = model(neg)

        loss = ebm_hardneg_loss(pos_e, neg_e)
        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()
        total += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total/len(loader):.4f}")

torch.save(
    accelerator.unwrap_model(model).state_dict(),
    "models/energy_head.pt",
)
