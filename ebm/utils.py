import torch
import torch.nn.functional as F


def ebm_inbatch_softmax_loss(energy_matrix: torch.Tensor):
    """
    energy_matrix: [B, B]
    Diagonal = positive
    """
    labels = torch.arange(
        energy_matrix.size(0),
        device=energy_matrix.device
    )
    loss = F.cross_entropy(-energy_matrix, labels)
    return loss


def collate_fn(batch):
    queries, positives = [], []

    for item in batch:
        q = item.get("query")
        passages = item.get("passages", {}).get("passage_text", [])
        labels = item.get("passages", {}).get("is_selected", [])

        if q is None:
            continue

        pos_idx = None
        for i, flag in enumerate(labels):
            if flag == 1:
                pos_idx = i
                break

        if pos_idx is None:
            continue

        queries.append(q)
        positives.append(passages[pos_idx])

    return queries, positives
