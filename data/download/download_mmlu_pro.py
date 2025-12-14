import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import logging
import shutil
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== 修改这里：新数据集信息 ======
DATASET_REPO = "TIGER-Lab/MMLU-Pro"

# 数据集在 HF repo 中的文件名及其对应的本地输出文件名
DOWNLOAD_FILES = {
    # HF 仓库文件路径: 目标本地文件名
    "data/test-00000-of-00001.parquet": "mmlu_pro_test.jsonl",
    "data/validation-00000-of-00001.parquet": "mmlu_pro_validation.jsonl",
}

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# 假设最终文件应保存在与脚本同目录的 'data' 目录中，就像您原始代码中的 'data/mmlu_pro_test.jsonl'
DATA_DIR = SCRIPT_DIR.parent  

def ensure_directories_exist():
    """确保目标保存目录存在。"""
    # 如果 DATA_DIR 已经是绝对路径，这将创建一个名为 'data' 的子目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"确保目录存在: {DATA_DIR}")

def convert_parquet_to_jsonl(parquet_path: Path, jsonl_path: Path):
    """
    读取 Parquet 文件，将其转换为 pandas DataFrame，然后保存为 jsonl 格式。
    """
    try:
        logger.info(f"Reading Parquet file: {parquet_path}")
        # 使用 pandas 读取 Parquet 文件
        df = pd.read_parquet(parquet_path)
        
        logger.info(f"Converting and saving to JSONL: {jsonl_path}")
        # 保存为 jsonl 格式。orient='records' 表示每行是一个完整的 JSON 对象。
        # lines=True 确保每条记录占一行。force_ascii=False 确保非 ASCII 字符（如中文）正常显示。
        df.to_json(
            jsonl_path, 
            orient='records', 
            lines=True, 
            force_ascii=False
        )
        logger.info(f"Saved {len(df)} records to {jsonl_path}")

    except Exception as e:
        logger.error(f"Failed to process and save {parquet_path} to {jsonl_path}: {e}")
        # 尝试删除可能已创建的不完整文件
        if jsonl_path.exists():
            jsonl_path.unlink()

def download_and_convert_mmlu_pro():
    """下载 MMLU-Pro 数据集文件并转换为 JSONL 格式。"""
    
    for hf_filename, local_filename in DOWNLOAD_FILES.items():
        output_file_path = DATA_DIR / local_filename
        
        try:
            logger.info(f"Downloading file: {hf_filename} from {DATASET_REPO}...")
            
            # 使用 hf_hub_download 下载文件到本地缓存
            file_path_in_cache = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=hf_filename,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN")
            )

            logger.info(f"Downloaded file to temporary location: {file_path_in_cache}")
            
            # 将下载的 Parquet 文件转换为 JSONL 格式并保存到目标路径
            convert_parquet_to_jsonl(Path(file_path_in_cache), output_file_path)

        except Exception as e:
            logger.error(f"Failed to download or process {hf_filename}: {e}")
            
def main():
    parser = argparse.ArgumentParser(description=f"Download and process MMLU-Pro dataset ({DATASET_REPO})")
    parser.add_argument("--token", help="Hugging Face token for accessing gated datasets")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 处理 token
    if args.token:
        os.environ["HF_TOKEN"] = args.token
        logger.info("Hugging Face token set")
    elif not os.environ.get("HF_TOKEN"):
        logger.warning("No Hugging Face token provided. You may encounter access issues.")
        
    ensure_directories_exist()
    
    # 执行下载和转换
    download_and_convert_mmlu_pro()

    logger.info("✅ MMLU-Pro dataset download and conversion completed!")

if __name__ == "__main__":
    main()