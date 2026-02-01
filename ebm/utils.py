# ebm/utils.py
import torch


def ebm_hardneg_loss(pos_e, neg_e, margin=1.0):
    """
    pos_e: [B]
    neg_e: [B, K]
    """
    return torch.relu(pos_e.unsqueeze(1) - neg_e + margin).mean()
