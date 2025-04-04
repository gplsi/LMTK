"""
Module: tokenization.utils
This module provides utility functions for building tokenization outputs used in causal
language modeling, as well as a state management class for tracking batch processing statuses.

Functions:
    - build_causal_lm_outputs: Converts tokenized lists to numpy arrays for efficient
      downstream processing.
    - build_causal_lm_outputs_old: Creates language modeling outputs using pre-allocated numpy
      arrays for potentially improved performance.

Classes:
    - StateManager: Manages processing states and asynchronous error handling.
"""

from asyncio import Queue
from typing import Dict, List
import numpy as np

def build_causal_lm_outputs(outputs: Dict[str, List]) -> Dict[str, np.ndarray]:
    """
    Build and return a dictionary of causal language modeling outputs.

    This function converts lists of token information into numpy arrays with a fixed integer type.
    The resulting dictionary includes:
        - 'input_ids': The token IDs converted to a numpy array.
        - 'attention_mask': The attention mask converted to a numpy array.
        - 'labels': Duplicates the input_ids array (commonly used as targets in causal LM tasks).

    Args:
        outputs (Dict[str, List]): A dictionary containing the following keys:
            - "input_ids": List of token ID lists.
            - "attention_mask": List of attention mask lists.

    Returns:
        Dict[str, np.ndarray]: A dictionary with numpy arrays for "input_ids", "attention_mask", and "labels".
    """
    
    return {
        "input_ids": np.asarray(outputs["input_ids"], dtype=np.int32),
        "attention_mask": np.asarray(outputs["attention_mask"], dtype=np.int32),
        "labels": np.asarray(outputs["input_ids"], dtype=np.int32)
    }

def build_causal_lm_outputs_old(outputs: Dict[str, List]) -> Dict[str, List]:
    """
    Build outputs for causal language modeling using pre-allocated arrays.

    This legacy function manually pre-allocates numpy arrays to store token IDs, attention masks,
    and language model labels for each sample in the batch. The approach involves:
        1. Determining batch size from the 'length' key.
        2. Pre-allocating arrays based on the batch size and the sequence length of the first example.
        3. Populating each array with corresponding data from the provided lists.
        4. Converting the numpy arrays back to lists before returning the dictionary.

    Args:
        outputs (Dict[str, List]): A dictionary expected to contain:
            - "input_ids": List of lists of token IDs.
            - "attention_mask": List of lists of attention masks.
            - "length": List with the sequence lengths for each example (assumes uniform sequence length).

    Returns:
        Dict[str, List]: A dictionary with processed lists for "input_ids", "attention_mask", and "labels".
    """
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
    """
    A class to manage state updates and asynchronous error handling during batch processing.

    Attributes:
        current_state (dict): A mapping of batch identifiers to their corresponding status.
        error_queue (asyncio.Queue): An asynchronous queue to manage and process errors.
    """
    def __init__(self):
        """
        Initialize the StateManager with an empty state signal dictionary and an error queue.
        """
        self.current_state = {}
        self.error_queue = Queue()
    
    def update_state(self, batch_id: str, status: str):
        """
        Update the processing state for a specific batch.

        This method sets or updates the status for the given batch identifier in the state dictionary,
        enabling tracking of the processing workflow at a granular level.

        Args:
            batch_id (str): Unique identifier for the processing batch.
            status (str): Current status or state message for the batch.

        Returns:
            None
        """
        self.current_state[batch_id] = status
