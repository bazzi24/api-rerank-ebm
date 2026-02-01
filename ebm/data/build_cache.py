import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def build_cache(
    model_name="sentence-transformers/msmarco-MiniLM-L6-v3",
    jsonl="data/msmarco_train.jsonl",
    out_pt="cache/hardneg_cache.pt",
    device="cuda",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    q_embs, p_embs, neg_embs = [], [], []

    with open(jsonl, encoding="utf-8") as f, torch.no_grad():
        for line in tqdm(f, desc="Building embedding cache"):
            item = json.loads(line)

            texts = [item["query"], item["positive"]] + item["negatives"]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            outputs = encoder(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu()

            q_embs.append(emb[0])
            p_embs.append(emb[1])
            neg_embs.append(emb[2:])

    data = {
        "query": torch.stack(q_embs),
        "positive": torch.stack(p_embs),
        "negatives": torch.stack(neg_embs),
    }

    torch.save(data, out_pt)
    print(f"✅ Saved cache to {out_pt}")
    print({k: v.shape for k, v in data.items()})


if __name__ == "__main__":
    build_cache()
