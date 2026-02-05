import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class JointEBMReranker(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        device: str = "cuda",
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.encoder.to(self.device)

        hidden = self.encoder.config.hidden_size

        self.energy_head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.SiLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        ).to(self.device)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode_pair(self, queries, docs):
        inputs = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        cls = outputs.last_hidden_state[:, 0]
        return cls

    def compute_energy_matrix(self, queries, docs):
        """
        Returns:
            energies: [B, B]
        """
        B = len(queries)
        q_rep = sum([[q] * B for q in queries], [])
        d_rep = docs * B

        emb = self.encode_pair(q_rep, d_rep)
        energy = self.energy_head(emb).view(B, B)
        return energy

    @torch.no_grad()
    def rerank(self, query, docs, top_k=5):
        queries = [query] * len(docs)
        emb = self.encode_pair(queries, docs)
        energies = self.energy_head(emb).squeeze(-1)

        idx = torch.argsort(energies)[:top_k]
        return [docs[i] for i in idx], energies[idx].cpu().tolist()
