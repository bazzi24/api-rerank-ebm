import yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm
from model import JointEBMReranker
from utils import ebm_margin_loss, collate_fn

# Load config
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

accelerator = Accelerator()

# Load dataset
print("Loading dataset...")
dataset = load_dataset(
    config['dataset']['name'],                        
    config['dataset']['config_name'],                 
    cache_dir=config['dataset']['cache_dir']
)
print(f"Dataset loaded: {len(dataset)} examples")
print("Dataset features:", dataset.features)

train_loader = DataLoader(
    dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# Model và optimizer
print("Initializing model...")
model = JointEBMReranker(config['model']['base_model'])
optimizer = AdamW(
    model.parameters(),
    lr=float(config['training']['learning_rate'])  # ← Ép kiểu thành float
)

model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# Train
model.train()
for epoch in range(config['training']['epochs']):
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")

    for batch_idx, (queries, positives, negatives) in enumerate(progress_bar):
        pos_energies, neg_energies = model(queries, positives, negatives)

        if neg_energies.numel() == 0:
            print(f"Warning: No negatives in batch {batch_idx}. Skipping.")
            continue

        loss = ebm_margin_loss(
            pos_energies,
            neg_energies,
            margin=config['training']['margin']
        )

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{config['training']['epochs']} - Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % config['training']['checkpoint_every'] == 0:
        accelerator.save_state(
            f"{config['training']['save_dir']}/checkpoint_epoch_{epoch+1}"
        )
        print(f"Checkpoint saved at epoch {epoch+1}")

# Save final model
unwrapped_model = accelerator.unwrap_model(model)
torch.save(
    unwrapped_model.state_dict(),
    f"{config['training']['save_dir']}/ebm_reranker_final.pt"
)
print("Training completed! Final model saved.")