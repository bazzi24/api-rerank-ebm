import torch
from tqdm import tqdm
from datasets import load_dataset
from ebm.model import JointEBMReranker


@torch.no_grad()
def evaluate(
    model,
    dataset,
    device="cuda",
    k_values=(1, 5, 10),
):
    model.eval()

    mrr = {k: 0.0 for k in k_values}
    recall = {k: 0.0 for k in k_values}
    total = 0

    for item in tqdm(dataset, desc="Evaluating"):
        query = item["query"]
        passages = item["passages"]["passage_text"]
        labels = item["passages"]["is_selected"]

        if 1 not in labels:
            continue

        # energy thấp = tốt
        queries = [query] * len(passages)
        energies = model.compute_energy_batch(queries, passages)

        sorted_idx = torch.argsort(energies).tolist()

        # rank của passage đúng đầu tiên
        rank = None
        for r, idx in enumerate(sorted_idx):
            if labels[idx] == 1:
                rank = r + 1  # rank bắt đầu từ 1
                break

        if rank is None:
            continue

        total += 1

        for k in k_values:
            if rank <= k:
                recall[k] += 1
                mrr[k] += 1.0 / rank

    for k in k_values:
        recall[k] /= max(total, 1)
        mrr[k] /= max(total, 1)

    return recall, mrr

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = JointEBMReranker(
        base_model_name=load_dataset("microsoft/ms_marco", "v2.1")["train"],
        device=device,
    )

    model.load_state_dict(torch.load("models/ebm_reranker_final.pt"))
    model.to(device)

    dataset = load_dataset(
        "microsoft/ms_marco",
        "v2.1",
        split="validation"
    )

    recall, mrr = evaluate(model, dataset)

    print("Recall:", recall)
    print("MRR:", mrr)

