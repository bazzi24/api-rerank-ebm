# test bi-encoder
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ================= CONFIG =================
MODEL_NAME = "sentence-transformers/msmarco-MiniLM-L6-v3"
INPUT_JSONL = "data/msmarco_train.jsonl"
OUT_CACHE = "cache/hardneg_cache.pt"

K_NEG = 4          
MAX_SAMPLES = None 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================


def build_cache():
    os.makedirs("cache", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    encoder.eval()

    q_embs = []
    p_embs = []
    neg_embs = []

    with open(INPUT_JSONL, "r", encoding="utf-8") as f, torch.no_grad():
        for idx, line in enumerate(tqdm(f, desc="Building embedding cache")):
            if MAX_SAMPLES and idx >= MAX_SAMPLES:
                break

            item = json.loads(line)

            negatives = item["negatives"]
            if len(negatives) < K_NEG:
                continue  

            negatives = negatives[:K_NEG]

            texts = [item["query"], item["positive"]] + negatives

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(DEVICE)

            outputs = encoder(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu()  # [2+K, D]

            q_embs.append(emb[0])                  # [D]
            p_embs.append(emb[1])                  # [D]
            neg_embs.append(emb[2:2 + K_NEG])      # [K, D]

    cache = {
        "query": torch.stack(q_embs),      # [N, D]
        "positive": torch.stack(p_embs),   # [N, D]
        "negatives": torch.stack(neg_embs) # [N, K, D]
    }

    torch.save(cache, OUT_CACHE)
    print(f"\n Cache saved to {OUT_CACHE}")
    print(f"Samples: {cache['query'].size(0)} | K_NEG={K_NEG}")


if __name__ == "__main__":
    build_cache()
