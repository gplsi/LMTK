from datasets import load_dataset
import argparse
import os

def download_dataset(dataset_name, output_path, column_name=None, column_name_value=None, split=None):
   
    # if split is none, it will return a datadict with all the splits
    if split is None:
        datasetdict = load_dataset(dataset_name, streaming=False)
        if column_name is not None and column_name_value is not None:
            for split_name, split_dataset in datasetdict.items():
                split_dataset = split_dataset.filter(lambda example: example[column_name] == column_name_value)
                datasetdict[split_name] = split_dataset
        
        # Save each split as separate parquet file
        os.makedirs(output_path, exist_ok=True)
        for split_name, split_dataset in datasetdict.items():
            split_dataset.to_parquet(f"{output_path}/{split_name}.parquet")
            
    else:
        dataset = load_dataset(dataset_name, streaming=False, split=split)
        if column_name is not None and column_name_value is not None:
            dataset = dataset.filter(lambda example: example[column_name] == column_name_value)
        dataset.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--column_name", type=str, required=False)
    parser.add_argument("--column_name_value", type=str, required=False)
    parser.add_argument("--split", type=str, required=False)
    args = parser.parse_args()
    
    download_dataset(args.dataset_name, args.output_path, args.column_name, args.column_name_value, args.split)
    print(f"Dataset saved to {args.output_path}")


