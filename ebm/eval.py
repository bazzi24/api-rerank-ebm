import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_mrr(
    model,
    dataset,
    device="cuda",
    k=10,
    max_samples=1000,
):
    """
    Sanity-check MRR@k cho MS MARCO v2.1
    - dùng subset nhỏ
    - chạy nhanh
    """

    model.eval()
    device = torch.device(device)

    mrr = 0.0
    total = 0

    for item in tqdm(dataset, desc=f"Eval MRR@{k}"):
        if total >= max_samples:
            break

        query = item["query"]
        passages = item["passages"]["passage_text"]
        labels = item["passages"]["is_selected"]

        if 1 not in labels:
            continue

        queries = [query] * len(passages)
        energies = model.compute_energy_batch(queries, passages)

        # energy thấp = tốt
        sorted_idx = torch.argsort(energies)

        rank = None
        for r, idx in enumerate(sorted_idx.tolist()):
            if labels[idx] == 1:
                rank = r + 1
                break

        if rank is None:
            continue

        if rank <= k:
            mrr += 1.0 / rank

        total += 1

    model.train()
    return mrr / max(total, 1)
