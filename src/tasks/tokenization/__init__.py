
# src/tasks/tokenization.py
def execute(config):
    print("Tokenizing with the following config:")
    print(config)
    
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