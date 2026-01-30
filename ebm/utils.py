import torch

def ebm_margin_loss(pos_energies, neg_energies, margin=1.0):
    
    loss = torch.relu(pos_energies - neg_energies + margin)
    return loss.mean()

def collate_fn(batch):
    
    queries = [item['anchor'] for item in batch]
    positives = [item['positive'] for item in batch]
    negatives = [[item['negative']] for item in batch]  
    
    return queries, positives, negatives