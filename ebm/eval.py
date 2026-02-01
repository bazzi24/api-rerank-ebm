import torch
from tqdm import tqdm
from ebm.model import JointEBMReranker


@torch.no_grad()
def evaluate(model, dataset, k_values=(1, 5, 10)):
    model.eval()
    recall = {k: 0 for k in k_values}
    mrr = {k: 0 for k in k_values}
    total = 0

    for item in tqdm(dataset):
        q = item["query"]
        passages = item["passages"]["passage_text"]
        labels = item["passages"]["is_selected"]

        if 1 not in labels:
            continue

        energies = model.compute_energy_matrix(
            [q], passages
        ).squeeze(0)

        idx = torch.argsort(energies)
        rank = None
        for r, i in enumerate(idx):
            if labels[i] == 1:
                rank = r + 1
                break

        if rank is None:
            continue

        total += 1
        for k in k_values:
            if rank <= k:
                recall[k] += 1
                mrr[k] += 1 / rank

    for k in k_values:
        recall[k] /= max(total, 1)
        mrr[k] /= max(total, 1)

    return recall, mrr
