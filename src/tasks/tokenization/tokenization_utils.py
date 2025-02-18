from datasets import Dataset
from typing import Optional



"""
The following script specifies the output format of the tokenized data according to different
training types i.e. causal language modeling, masked language modeling, etc.
"""



### TOKENIZED FORMAT HANDLERS ###

def causal_pretraining_format(outputs: dict[str]) -> dict[str]: 
    input_batch = []
    att_mask_batch = []
    labels_batch=[]
    for length, input_ids, attention_mask in zip(outputs["length"], outputs["input_ids"], outputs["attention_mask"]):
        input_batch.append(input_ids)
        att_mask_batch.append(attention_mask)
        labels = input_ids
        labels_batch.append(labels)  
        
    # Ensure output for every key
    assert len(input_batch) == len(att_mask_batch) == len(labels_batch), "Mismatch in output lengths."
    
    return {
        "input_ids": input_batch,
        "attention_mask": att_mask_batch,
        "labels": labels_batch,
    }
    
    
    
# General mapper for tokenized format handlers
TOKENIZED_FORMAT_HANDLER = {
    'causal_pretraining': causal_pretraining_format,   
}



### TOKENIZE FUNCTIONS ###

def causal_pretraining_tokenizing_function(tokenizer, batch: Dataset, context_length: int, overlap: int) -> dict[str]:
        
    """
    The function tokenizes the input text data for continual pretraining tasks.
    This function is designed to truncate the text data into a fixed length based on context_length.
    The truncated elements will be returned in the subsequent batches with an stride equal to the given overlap.
    """
    
    outputs = tokenizer(
        batch["content"],
        truncation=True,
        max_length=context_length,  # Do not truncate here to handle splitting manually
        return_overflowing_tokens=True,
        return_length=True,
        stride=overlap,
        padding=True
    )
    output_format = TOKENIZED_FORMAT_HANDLER['causal_pretraining'](outputs)
    return output_format




# General mapper for tokenize functions

TOKENIZE_FUNCTION_HANDLER = {
    'causal_pretraining': causal_pretraining_tokenizing_function,
}


