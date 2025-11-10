#!/bin/bash
# This script activates the virtual environment and downloads the GAIA dataset.

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo "Activated virtual environment."
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Please set it before running the script: export HF_TOKEN='your_token_here'"
  exit 1
fi

echo "Installing required packages for dataset download..."
pip install huggingface_hub tqdm requests

echo "====================================================="
echo "Starting GAIA dataset download..."
echo "====================================================="

python data/download/download_gaia.py --token "$HF_TOKEN"

echo "====================================================="
echo "Download script finished."
echo "====================================================="