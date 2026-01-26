#!/bin/bash

# Path to the config file for kViT_prostate
CONFIG_PATH="src/configs/train_image_2D.yaml"
OUTPUT_BASE="src/output"
MODEL_NAME="train_0.0001_best_prostaterun_resnet"
MODEL_PATH="$OUTPUT_BASE/$MODEL_NAME"
TRAIN_CSV="/data/fastmri_prostate/2d_t2_train.csv"
TEST_CSV="/data/fastmri_prostate/2d_t2_test.csv"

# Create config file
cat <<EOL > "$CONFIG_PATH"
# Configuration file for unified training script - MIL dataset mode
# This config is for training with Multiple Instance Learning (MIL) approach

csv_path: $TRAIN_CSV      # CSV file with data information
in_channel: 1       # Number of input channels (e.g. 1 for grey-scale, 3 for rgb).
num_classes: 2      # Number of output channels.
crop_size: 224      # Size of the crop to be used for training.
base_output: $OUTPUT_BASE       # Base output path for saving of results.
lr: 0.0001
optimizer: AdamW
batch_size: 128      # Batch size
num_workers: 8      # Workers used by dataloader
num_threads: 1      # Number of threads used by the script
pin_memory: true
# Dataset configuration
dataset_type: 2D    # Options: "2D" or "MIL" - determines which model and data loading approach to use
dataset_name: FastMriDataset_Prostate  # Options: "FastMriDataset_Knee" "FastMriDataset_Prostate", "ImageDataset", "kSpaceDataset" (for MIL mode)
kspace: false       # Whether to use k-space data (only applicable for 2D mode)
hybrid: false       # Whether to use hybrid approach (only applicable for 2D mode)
model: resnet          # Model to be used. Options: "vit", "efficientnet", or "resnet"

comment: best_prostaterun_resnet  # Freeform to give additional context to saving file.
EOL

# Train the model
python src/train_image.py --config "$CONFIG_PATH"

# Test the trained model on all undersampling rates
for u in 0 2 4 6 8 10 12 16 24; do
    python src/test_model_2D.py \
        -m "$MODEL_PATH" \
        -c "$TEST_CSV" \
        -u "$u" \
        -o "src/output/test_results/${MODEL_NAME}"
done
