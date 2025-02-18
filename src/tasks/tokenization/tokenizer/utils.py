# src/tokenization/utils.py
from typing import Dict, List
import numpy as np

def build_causal_lm_outputs(outputs: Dict[str, List]) -> Dict[str, List]:
    """Build outputs for causal language modeling."""
    batch_size = len(outputs["length"])
    
    # Pre-allocate arrays for better performance
    input_ids = np.zeros((batch_size, outputs["length"][0]), dtype=np.int32)
    attention_mask = np.zeros_like(input_ids)
    labels = np.zeros_like(input_ids)
    
    for idx in range(batch_size):
        input_ids[idx] = outputs["input_ids"][idx]
        attention_mask[idx] = outputs["attention_mask"][idx]
        labels[idx] = outputs["input_ids"][idx]
    
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist()
    }
