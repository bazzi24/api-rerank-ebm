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

        # ===== Encoder =====
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.encoder.to(self.device)

        hidden = self.encoder.config.hidden_size

        # ===== Energy Head (IMPORTANT) =====
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

    # ------------------------------------------------
    # Encode text -> CLS embedding
    # ------------------------------------------------
    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.encoder(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return cls_emb

    # ------------------------------------------------
    # Compute energy for batch
    # ------------------------------------------------
    def compute_energy_batch(self, queries: list[str], docs: list[str]) -> torch.Tensor:
        assert len(queries) == len(docs)

        texts = [f"{q} [SEP] {d}" for q, d in zip(queries, docs)]
        emb = self.encode(texts)
        energy = self.energy_head(emb).squeeze(-1)
        return energy

    # ------------------------------------------------
    # Training forward (in-batch negatives)
    # ------------------------------------------------
    def forward(self, queries: list[str], positives: list[str]):
        """
        Returns:
            energies: shape [B]  (positive energies)
        """
        energies = self.compute_energy_batch(queries, positives)
        return energies

    # ------------------------------------------------
    # Inference rerank (for RAGFlow)
    # ------------------------------------------------
    @torch.no_grad()
    def rerank(self, query: str, docs: list[str], top_k: int = 5):
        queries = [query] * len(docs)
        energies = self.compute_energy_batch(queries, docs)

        sorted_idx = torch.argsort(energies)
        top_idx = sorted_idx[:top_k]

        reranked_docs = [docs[i] for i in top_idx]
        reranked_energies = energies[top_idx].cpu().tolist()

        return reranked_docs, reranked_energies
