from datasets import load_dataset

dataset = load_dataset("ms_marco", "v2.1")

print(dataset)

print("=======================")

print(dataset["train"].features)