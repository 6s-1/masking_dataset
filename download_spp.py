from datasets import load_dataset

# Load the dataset
ds = load_dataset("wuyetao/spp")

# Save the train split locally in JSONL format
ds['train'].to_json("spp_train.jsonl")

print("Dataset saved as spp_train.jsonl ✅")
