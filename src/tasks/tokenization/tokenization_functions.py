from tokenized_format import *
from base_tokenization import *
"""
This script contains functions for tokenizing text data for language modeling tasks.
"""

class Tokenize(BaseTokenizer):
    
    """
    This is the main class for tokenizing, it will perform specific tokenization based on the desired task.
    Currently supports values in ['continual_pretraining']. kwargs expects the specific parameters for the 
    tokenization task.
    """
    def __init__(
        self,
        task: str,
        **kwargs,
    ):
        self.task = task
        self.kwargs = kwargs
        
        if self.task == 'continual_pretraining':
            self.context_length = self.kwargs['context_length']
            self.overlap = kwargs['overlap']
        else:
            raise ValueError(f"Task {self.task} is not supported.")
        super().__init__(**kwargs)
        
    def tokenize_dataset(self, dataset: Dataset):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define the dataset features
        features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
        })
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            lambda batch: TOKENIZE_FUNCTION_HANDLER[self.task](tokenizer, batch, context_length=self.context_length, overlap=self.overlap),
            batched=True, features=features)
        
        return tokenized_dataset


def continual_pretraining_tokenizing_function(tokenizer, batch, context_length, overlap):
        
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

TOKENIZE_FUNCTION_HANDLER = {
    'continual_pretraining': continual_pretraining_tokenizing_function,
}