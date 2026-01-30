# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer



class JointEBMReranker(nn.Module):
    def __init__(self, base_model_name='sentence-transformers/all-MiniLM-L6-v2', device='cuda'):
        """
        
        
        Args:
            base_model_name (str): Tên model từ Hugging Face (ví dụ: 'sentence-transformers/all-MiniLM-L6-v2')
            device (str): 'cuda' hoặc 'cpu'
        """
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        # Load encoder và tokenizer
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Chuyển model lên device ngay
        self.encoder.to(self.device)
        
        hidden_size = self.encoder.config.hidden_size
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)  

    def compute_energy(self, query: str, doc: str, training: bool = True):
        """
        Tính energy cho một cặp (query, doc).
        
        Args:
            query (str): Câu hỏi/query
            doc (str): Document/passage
            training (bool): Nếu False thì dùng no_grad để tiết kiệm bộ nhớ
        
        Returns:
            torch.Tensor: Energy scalar
        """
        text = f"{query} [SEP] {doc}"
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if not training:
            with torch.no_grad():
                outputs = self.encoder(**inputs)
        else:
            outputs = self.encoder(**inputs)

        pooled = outputs.last_hidden_state[:, 0, :]  # Lấy [CLS] token
        energy = self.energy_head(pooled).squeeze(-1)  # scalar
        return energy

    def forward(self, queries, positives, negatives):
        """
        Forward cho training: tính energy cho positives và negatives.
        
        Args:
            queries (list[str]): List các query
            positives (list[str]): List các positive document
            negatives (list[list[str]]): List các list negative documents
        
        Returns:
            tuple: (pos_energies, neg_energies)
        """
        # Tính positive energies
        pos_energies = []
        for q, p in zip(queries, positives):
            pos_energies.append(self.compute_energy(q, p, training=self.training))
        pos_energies = torch.stack(pos_energies)

        # Tính negative energies (giới hạn 5 neg/query để tránh OOM)
        neg_energies = []
        for q, neg_list in zip(queries, negatives):
            for neg in neg_list:
                neg_energies.append(self.compute_energy(q, neg, training=self.training))
        if neg_energies:  # tránh stack rỗng
            neg_energies = torch.stack(neg_energies)
        else:
            neg_energies = torch.tensor([]).to(self.device)

        return pos_energies, neg_energies

    @torch.no_grad()
    def rerank(self, query: str, docs: list[str], top_k: int = 5):
        """
        Hàm tiện ích để rerank trong inference (không cần gradient).
        
        Args:
            query (str): Query
            docs (list[str]): List documents cần rerank
            top_k (int): Số document trả về
        
        Returns:
            tuple: (reranked_docs, energies)
        """
        energies = []
        for doc in docs:
            energy = self.compute_energy(query, doc, training=False)
            energies.append(energy.item())

        # Sort: energy thấp nhất = relevant nhất
        sorted_indices = sorted(range(len(energies)), key=lambda i: energies[i])
        reranked_docs = [docs[i] for i in sorted_indices[:top_k]]
        reranked_energies = [energies[i] for i in sorted_indices[:top_k]]

        return reranked_docs, reranked_energies