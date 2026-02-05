import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def build_hardneg_cache(
    model_name,
    jsonl_file,
    device="cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    q_embs, p_embs, neg_embs = [], [], []

    with open(jsonl_file) as f, torch.no_grad():
        for line in tqdm(f):
            item = json.loads(line)

            texts = (
                [item["query"], item["positive"]] +
                item["negatives"]
            )

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            emb = encoder(**inputs).last_hidden_state[:, 0, :].cpu()

            q_embs.append(emb[0])
            p_embs.append(emb[1])
            neg_embs.append(emb[2:])

    return (
        torch.stack(q_embs),
        torch.stack(p_embs),
        torch.stack(neg_embs),
    )
