from argparse import ArgumentParser


def parse_args(sequence_length, model_name, input_dir):
    
    
    dataset_dir =  "splitted_datasets_fabio"
    arrow_dataset_dir = "arrow_datasets_fabio"
    tokenized_dataset_dir = "tokenized_datasets_fabio"
    parser = ArgumentParser(description="Tokenize and filter a dataset for causal language modeling.")
    
    
    dataset_name = model_name.split("/")[-1]
    dataset_name = dataset_name + "_dataset"

    # SAVED FOLDERS
    parser.add_argument("--initial_dataset_dir", type=str, help="Name of the dataset original folder.", default=input_dir)
    parser.add_argument("--dataset_dir", type=str, help="Name of the dataset already splitted.", default=dataset_dir)
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to tokenize and filter.", default=dataset_name)
    parser.add_argument("--arrow_dataset_dir", type=str, help="Name of the arrow huggingface dataset.", default=arrow_dataset_dir)
    parser.add_argument("--tokenized_dataset_dir", type=str, help="Name of the tokenized dataset.", default=tokenized_dataset_dir)


    # SPLITTING INITIAL FILES PARAMETERS
    parser.add_argument("--max_file_size", type=int, help="Maximum size of the split files in bytes.", default=100 * 1024 * 1024)
    

    # HUGGINGFACE DATASET PARAMETERS
    parser.add_argument("--test_split", type=float, help="Fraction of the dataset to use for testing.", default=0.05)
    


    # TOKENIZATION PARAMETERS
    parser.add_argument("--tokenizer_name", type=str, help="Name of the tokenizer to use.", default=model_name)
    parser.add_argument("--model_name", type=str, help="Name of the model to use for tokenization.", default=model_name)
    parser.add_argument("--max_length", type=int, help="Maximum length of the tokenized sequences.", default=sequence_length)
    parser.add_argument("--overlap", type=int, help="Overlap between tokenized sequences.", default=sequence_length // 4)

    args = parser.parse_args()

    args_dict = vars(args)
    return args_dict
