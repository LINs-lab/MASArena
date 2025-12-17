"""
This module is used to download the dataset from the internet.
"""

import os
import argparse
import logging
import json
from pathlib import Path
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

try:
    from datasets import load_dataset
    from datasets import get_dataset_config_names
except ImportError:
    load_dataset = None
    get_dataset_config_names = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "dataset_config.json"

def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {CONFIG_PATH}: {e}")
            return {}
    else:
        logger.warning(f"Config file not found at {CONFIG_PATH}")
        return {}

# Mapping from dataset name to Hugging Face repository details
# Keys should match the benchmark names used in the system
# Users should fill in the 'repo_id' and 'filename' (if different from local)
DATASET_MAPPING = load_config()

def download_dataset(dataset_name: str):
    """
    Download a specific dataset from Hugging Face.
    
    Args:
        dataset_name: The name of the dataset to download (key in DATASET_MAPPING)
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    if dataset_name not in DATASET_MAPPING:
        logger.error(f"Dataset '{dataset_name}' not found in mapping.")
        return False
    
    config = DATASET_MAPPING[dataset_name]
    repo_id = config.get("repo_id")
    filename = config.get("filename")
    local_name = config.get("local_name")
    
    if not repo_id:
        logger.warning(f"Repo ID not configured for dataset '{dataset_name}'. Skipping.")
        return False

    # Check if filename is null, which indicates we should use datasets.load_dataset
    if filename is None:
        if load_dataset is None:
            logger.error("datasets library is not installed. Please install it with 'pip install datasets'.")
            return False
            
        try:
            logger.info(f"Downloading dataset {dataset_name} from {repo_id} using datasets library...")
            
            # For BBH specifically, we need to handle configs
            if "bbh" in repo_id.lower():
                try:
                    configs = get_dataset_config_names(repo_id)
                except Exception as e:
                    logger.warning(f"Could not get configs for {repo_id}: {e}. Trying default load.")
                    configs = None

                all_data = []
                if configs:
                    logger.info(f"Found {len(configs)} configs for {repo_id}. Downloading each...")
                    for config_name in configs:
                        logger.info(f"Downloading config: {config_name}")
                        try:
                            # Load specific config
                            ds = load_dataset(repo_id, config_name)
                            # Iterate through splits (usually 'test' for BBH)
                            for split in ds.keys():
                                subset = ds[split]
                                for item in subset:
                                    # Add config name to item for reference if needed
                                    item['subset'] = config_name
                                    all_data.append(item)
                        except Exception as e:
                            logger.error(f"Failed to download config {config_name}: {e}")
                else:
                    # Try loading without config
                    ds = load_dataset(repo_id)
                    if hasattr(ds, 'keys'):
                        for key in ds.keys():
                            subset = ds[key]
                            for item in subset:
                                all_data.append(item)
                    else:
                        for item in ds:
                            all_data.append(item)

            else:
                # Generic load for other datasets
                ds = load_dataset(repo_id)
                all_data = []
                if hasattr(ds, 'keys'):
                    for key in ds.keys():
                        subset = ds[key]
                        for item in subset:
                            all_data.append(item)
                else:
                    for item in ds:
                        all_data.append(item)
            
            # Target path
            target_path = DATA_DIR / local_name
            
            # Ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Write to JSONL
            logger.info(f"Writing {len(all_data)} items to {target_path}...")
            with open(target_path, 'w', encoding='utf-8') as f:
                for item in all_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"Successfully processed {dataset_name} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download/process {dataset_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    else:
        # Standard file download using hf_hub_download
        if hf_hub_download is None:
            logger.error("huggingface_hub library is not installed. Please install it with 'pip install huggingface_hub'.")
            return False
            
        try:
            logger.info(f"Downloading {dataset_name} from {repo_id}/{filename}...")
            
            # Download to cache first
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN")
            )
            
            # Move/Copy to data directory
            target_path = DATA_DIR / local_name
            
            # Ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Copy content
            import shutil
            shutil.copy2(downloaded_path, target_path)
            
            logger.info(f"Successfully downloaded {dataset_name} to {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False

def download_all():
    """Download all configured datasets."""
    logger.info("Starting download of all datasets...")
    success_count = 0
    for name in DATASET_MAPPING:
        if download_dataset(name):
            success_count += 1
    logger.info(f"Finished downloading all datasets. Successful: {success_count}/{len(DATASET_MAPPING)}")

def ensure_dataset_exists(dataset_name: str, data_path: str = None):
    """
    Check if dataset exists, and if not, try to download it.
    This function is intended to be called by the benchmark runner.
    
    Args:
        dataset_name: The name of the benchmark/dataset
        data_path: Optional specific path to check. If None, checks the default location.
    """
    if data_path:
        path = Path(data_path)
    else:
        if dataset_name in DATASET_MAPPING:
            path = DATA_DIR / DATASET_MAPPING[dataset_name]["local_name"]
        else:
            # Fallback for unknown datasets that follow the naming convention
            path = DATA_DIR / f"{dataset_name}_test.jsonl"
            
    if path.exists():
        return True
        
    logger.info(f"Dataset file not found at {path}. Attempting to download...")
    
    # Map dataset_name to our keys if possible
    if dataset_name in DATASET_MAPPING:
        return download_dataset(dataset_name)
    else:
        logger.warning(f"Dataset '{dataset_name}' not known in download configuration. Cannot auto-download.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")
    parser.add_argument("dataset", nargs="?", default="all", 
                        help="Name of the dataset to download (e.g., 'math', 'gsm8k') or 'all'")
    
    args = parser.parse_args()
    
    if args.dataset.lower() == "all":
        download_all()
    else:
        download_dataset(args.dataset)

if __name__ == "__main__":
    main()
