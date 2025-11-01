from datasets import load_dataset
import argparse
#  Script for downloading and filtering a dataset based on specific column values.
#  Useful for dealing with the datasets clearly splitted in different languages.
#  Recommended for saving the input datasets before tokenization with LMTK.


def download_dataset(dataset_name, output_path, column_name=None, column_name_value=None, split=None):
   
   # if split is none, it will return a datadict with all the splits
   if split is None:
      datasetdict = load_dataset(dataset_name, streaming=False)
      if column_name is not None and column_name_value is not None:
        for split_name, split_dataset in datasetdict.items():
          split_dataset = split_dataset.filter(lambda example: example[column_name] == column_name_value)
          datasetdict[split_name] = split_dataset
      datasetdict.save_to_disk(output_path)
   else:
      dataset = load_dataset(dataset_name, streaming=False, split=split)
      if column_name is not None and column_name_value is not None:
        dataset = dataset.filter(lambda example: example[column_name] == column_name_value)
      dataset.save_to_disk(output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--column_name", type=str, required=False)
    parser.add_argument("--column_name_value", type=str, required=False)
    parser.add_argument("--split", type=str, required=False)
    args = parser.parse_args()
    download_dataset(args.dataset_name, args.output_path, args.column_name, args.column_name_value, args.split)

