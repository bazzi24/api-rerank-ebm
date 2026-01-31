from datasets import load_dataset

dataset = load_dataset("ms_marco", "v1.1")

print(len(dataset))