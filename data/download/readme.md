# GAIA Dataset Downloader

This script downloads the GAIA benchmark dataset from Hugging Face.

## Prerequisites

1. Install required packages:
```bash
pip install huggingface_hub tqdm requests
```

2. You need a Hugging Face account and token with access to the GAIA dataset.
   - Create an account at https://huggingface.co/
   - Generate a token at https://huggingface.co/settings/tokens
   - Accept the terms and conditions for the GAIA dataset at https://huggingface.co/datasets/gaia-benchmark/GAIA

## Usage

```bash
# Download both test and validation datasets
python download_gaia.py --token YOUR_HUGGING_FACE_TOKEN

# Download only test dataset
python download_gaia.py --split test --token YOUR_HUGGING_FACE_TOKEN

# Download only validation dataset
python download_gaia.py --split validation --token YOUR_HUGGING_FACE_TOKEN
```

## Output

The script will:
1. Download metadata files to:
   - `data/gaia_test.jsonl`
   - `data/gaia_validate.jsonl`

2. Download associated files to:
   - `data/files/gaia/test/`
   - `data/files/gaia/validate/`

## Note

The GAIA dataset is gated, which means you must have appropriate permissions to access it. If you encounter access issues, make sure you've accepted the dataset terms on the Hugging Face website. 