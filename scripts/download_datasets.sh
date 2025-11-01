#!/bin/bash

# Script for batch downloading and filtering HuggingFace datasets
# Make sure your Python script is in the same directory or provide the full path

PYTHON_SCRIPT="./scripts/download_dataset.py"  # Change this to your script's path if different
BASE_OUTPUT_DIR="./data/"  # Base directory for all downloaded datasets

# Create base output directory
mkdir -p $BASE_OUTPUT_DIR

echo "Starting batch dataset download and filtering..."

echo "Downloading ALIA DOGV VA train"         # DC8
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_dogv" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_dogv_train_va" \
    --column_name "language" \
    --column_name_value "va"


echo "Downloading ALIA DOGV ES train"         # DC9
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_dogv" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_dogv_train_es" \
    --column_name "language" \
    --column_name_value "es"


echo "Downloading ALIA LES CORTS VA|ES train"         # DC10
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_les_corts" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_les_corts_train_va" \
    --column_name "language" \
    --column_name_value "va|es"


echo "Downloading ALIA AMIC VA train"                 # DC11
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_amic" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_amic_train_va" \
    --column_name "language" \
    --column_name_value "va"


echo "Downloading ALIA BOUA VA train"                 # DC12
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_boua" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_boua_train_va" \
    --column_name "language" \
    --column_name_value "va"

echo "Downloading ALIA BOUA ES train"                 # DC13
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_boua" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_boua_train_es" \
    --column_name "language" \
    --column_name_value "es"

echo "Downloading ALIA TOURISM VA train"                 # DC14
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_tourism" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_tourism_train_va" \
    --column_name "language" \
    --column_name_value "va"


echo "Downloading ALIA TOURISM ES train"                 # DC15
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_tourism" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_tourism_train_es" \
    --column_name "language" \
    --column_name_value "es"


echo "Downloading ALIA TOURISM EN train"                 # DC16
python $PYTHON_SCRIPT \
    --dataset_name "gplsi/alia_tourism" \
    --split "train" \
    --output_path "$BASE_OUTPUT_DIR/alia_tourism_train_en" \
    --column_name "language" \
    --column_name_value "en"


echo "Batch download completed!"
echo "All datasets saved in: $BASE_OUTPUT_DIR"
