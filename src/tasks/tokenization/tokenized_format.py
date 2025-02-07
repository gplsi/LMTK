from typing import Optional
"""
The following script specifies the output format of the tokenized data according to different
training types i.e. causal language modeling, masked language modeling, etc.
"""

def causal_pretraining_format(outputs: dict[str]):
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
    
    
    
    
TOKENIZED_FORMAT_HANDLER = {
    'causal_pretraining': causal_pretraining_format,   
}