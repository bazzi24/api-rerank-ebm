import torch
from ebm.model import JointEBMReranker

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

model = JointEBMReranker(
    base_model_name='sentence-transformers/all-MiniLM-L6-v2',
    device=device
)


try:
    model.load_state_dict(torch.load('models/ebm_reranker_final.pt', map_location=device))
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

model.to(device)
model.eval()

def rerank(query: str, docs: list[str], top_k: int = 5):
    """
    Rerank danh sách documents dựa trên energy từ EBM model.
    
    Args:
        query (str): Câu hỏi/query
        docs (list[str]): Danh sách documents cần rerank
        top_k (int): Số documents trả về (default: 5)
    
    Returns:
        tuple: (reranked_docs, reranked_energies)
    """
    # Sử dụng hàm rerank đã có sẵn trong model (đã tối ưu với @torch.no_grad())
    reranked_docs, reranked_energies = model.rerank(query, docs, top_k=top_k)
    return reranked_docs, reranked_energies

### TEST
if __name__ == "__main__":
    query = "What is the capital of France?"
    
    sample_docs = [
        "Paris is the capital city of France.",
        "France is in Europe.",
        "The Eiffel Tower is in Paris.",
        "Berlin is the capital of Germany.",
        "Python is a programming language.",
        "Khoa Truong Dinh is a Data Engineering",
        "Bazzi Github is popular",
        "Monaco is a city of France"
    ]
    
    print(f"\nQuery: {query}")
    print("Original documents:")
    for i, doc in enumerate(sample_docs, 1):
        print(f"  {i}. {doc}")
    
    print("\nReranked results:")
    reranked, scores = rerank(query, sample_docs, top_k=5)
    
    for i, (doc, score) in enumerate(zip(reranked, scores), 1):
        print(f"  {i}. Energy: {score:.4f} | {doc}")