# test
import json
from tqdm import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi


def mine_hard_negatives(
    split="train",
    top_k=20,
    out_file="hard_negatives.jsonl",
):
    dataset = load_dataset("microsoft/ms_marco", "v2.1")[split]

    corpus = []
    corpus_meta = []

    for item in dataset:
        for text in item["passages"]["passage_text"]:
            corpus.append(text)
            corpus_meta.append(item["query"])

    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    with open(out_file, "w") as f:
        for item in tqdm(dataset):
            query = item["query"]
            passages = item["passages"]["passage_text"]
            labels = item["passages"]["is_selected"]

            if 1 not in labels:
                continue

            pos_idx = labels.index(1)
            pos_text = passages[pos_idx]

            scores = bm25.get_scores(query.split())
            ranked = sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )

            hard_negs = []
            for idx, _ in ranked:
                cand = corpus[idx]
                if cand != pos_text:
                    hard_negs.append(cand)
                if len(hard_negs) >= top_k:
                    break

            record = {
                "query": query,
                "positive": pos_text,
                "negatives": hard_negs,
            }

            f.write(json.dumps(record) + "\n")

    print(f"Saved hard negatives → {out_file}")
