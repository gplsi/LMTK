import os
from datasets import load_from_disk, DatasetDict, Dataset
import torch

# Path to the tokenized dataset
dataset_path = "/workspace/output/tokenized_gpt2_instructions"

# Load the dataset from disk
dataset = load_from_disk(dataset_path)

# Check if it's a DatasetDict (splits) or a single Dataset
if isinstance(dataset, DatasetDict):
    print(f"Loaded dataset splits: {list(dataset.keys())}")
    valid_split = None
    for split in dataset.keys():
        if 'valid' in split or 'val' in split or 'test' in split:
            valid_split = split
            break
    if valid_split is None:
        valid_split = list(dataset.keys())[0]
    print(f"Using split: {valid_split}")
    ds = dataset[valid_split]
else:
    print("Loaded a single dataset (no splits)")
    ds = dataset

print(f"Number of samples: {len(ds)}")

# Show the format and a sample
sample = ds[:100]  # Get first 100 samples as a batch
print("Sample keys:", list(sample.keys()))
for k in sample:
    v = sample[k]
    print(f"{k}: type={type(v)}, shape={getattr(v, 'shape', None)}")
    # Convert to tensor if it's a list of lists (common for HuggingFace datasets)
    if isinstance(v, list) and isinstance(v[0], list):
        v_tensor = torch.tensor(v)
        print(f"  dtype: {v_tensor.dtype}, shape: {v_tensor.shape}")
        print(f"  min: {v_tensor.min().item()}, max: {v_tensor.max().item()}")
        print(f"  first row: {v_tensor[0][:20]}")
        print(f"  last row: {v_tensor[-1][:20]}")
    elif isinstance(v, torch.Tensor):
        print(f"  dtype: {v.dtype}, values: {v[:10]}")
        print(f"  min: {v.min().item()}, max: {v.max().item()}")
        print(f"  first row: {v[0][:20]}")
        print(f"  last row: {v[-1][:20]}")
    elif isinstance(v, list):
        print(f"  first 10 values: {v[:10]}")
