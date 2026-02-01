import json
from datasets import load_dataset
from tqdm import tqdm


def convert_msmarco(
    out_file="data/msmarco_train.jsonl",
    max_negatives=4,
    max_samples=None,
):
    ds = load_dataset("microsoft/ms_marco", "v2.1")["train"]

    written = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="Converting MS MARCO"):
            query = item.get("query")
            passages = item.get("passages", {})
            texts = passages.get("passage_text", [])
            labels = passages.get("is_selected", [])

            if query is None or not texts:
                continue

            pos_idx = None
            negs = []

            for i, flag in enumerate(labels):
                if flag == 1 and pos_idx is None:
                    pos_idx = i
                elif flag == 0:
                    negs.append(texts[i])

            if pos_idx is None or len(negs) == 0:
                continue

            record = {
                "query": query,
                "positive": texts[pos_idx],
                "negatives": negs[:max_negatives],
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if max_samples and written >= max_samples:
                break

    print(f"Wrote {written} samples to {out_file}")


if __name__ == "__main__":
    convert_msmarco(
        out_file="data/msmarco_train.jsonl",
        max_negatives=4,
        max_samples=None,  # set số nhỏ nếu test
    )
