import torch


def ebm_inbatch_loss(energies: torch.Tensor, margin: float = 1.0):
    """
    energies: shape [B]  (positive energies)
    """
    B = energies.size(0)

    pos = energies.unsqueeze(1)        # [B, 1]
    neg = energies.unsqueeze(0)        # [1, B]

    loss = torch.relu(pos - neg + margin)

    # remove self-comparison
    mask = torch.eye(B, device=energies.device)
    loss = loss * (1 - mask)

    return loss.mean()

def collate_fn(batch):
    """
    Robust collate function cho MS MARCO v2.1

    Output:
      - queries:   List[str]
      - positives: List[str]

    Quy tắc:
      - Mỗi query lấy 1 positive passage đầu tiên (is_selected == 1)
      - Nếu sample không có positive → skip
    """

    queries = []
    positives = []

    for item in batch:
        query = item.get("query", None)
        passages = item.get("passages", {}).get("passage_text", [])
        labels = item.get("passages", {}).get("is_selected", [])

        if query is None or not passages or not labels:
            continue

        # tìm positive
        pos_idx = None
        for i, flag in enumerate(labels):
            if flag == 1:
                pos_idx = i
                break

        if pos_idx is None:
            continue

        queries.append(query)
        positives.append(passages[pos_idx])

    return queries, positives


