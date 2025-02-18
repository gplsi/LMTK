import pprint

from box import Box

# src/tasks/tokenization.py
def execute(config: Box):
    print("Tokenizing with the following config:")
    pprint.pprint(config.items)
    
    # input dataset
    # - huggingface
    # - local 
    #   - transformers dataset
    #   - files
    #       - csv
    #       - txt
    #       - json
    
    # workflow
    # if dataset from files and not transformer dataset
    # - load from files
    # - create dataset
    # - save to disk (if specified)
    
    # then for all datasets
    # - tokenize
    # - save to disk (if specified)