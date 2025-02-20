# src/tokenization/utils.py
from asyncio import Queue
from typing import Dict, List
import numpy as np

def build_causal_lm_outputs(outputs: Dict[str, List]) -> Dict[str, List]:
    return {
        "input_ids": np.asarray(outputs["input_ids"], dtype=np.int32),
        "attention_mask": np.asarray(outputs["attention_mask"], dtype=np.int32),
        "labels": np.asarray(outputs["input_ids"], dtype=np.int32)
    }

def build_causal_lm_outputs_old(outputs: Dict[str, List]) -> Dict[str, List]:
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

class StateManager:
    def __init__(self):
        self.current_state = {}
        self.error_queue = Queue()
    
    def update_state(self, batch_id: str, status: str):
        self.current_state[batch_id] = status
