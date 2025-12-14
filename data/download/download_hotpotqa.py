import os
import argparse
from pathlib import Path
from tqdm import tqdm # 尽管在这个脚本中可能用不到，但为了保持风格可以保留
import pandas as pd
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_REPO = "hotpotqa/hotpot_qa"
# HotpotQA 的 fullwiki 测试集文件
PARQUET_FILENAME = "distractor/validation-00000-of-00001.parquet"

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# 目标：将最终的 jsonl 文件放在脚本所在目录下的 'data' 文件夹中
DATA_DIR = SCRIPT_DIR.parent 
# 最终保存的 metadata 文件路径
HOTPOTQA_TEST_OUTPUT = DATA_DIR / "hotpotqa_test.jsonl" 

def ensure_directories_exist():
    """确保输出目录 DATA_DIR 存在。"""
    # 创建 DATA_DIR 目录（包括父目录），如果已存在则不报错 
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured data directory exists: {DATA_DIR}")

def parquet_to_jsonl(parquet_file: Path, jsonl_file: Path):
    """
    将 Parquet 文件转换为 JSONL 格式。
    """
    logger.info(f"Converting {parquet_file.name} to JSONL format...")
    try:
        # 读取 Parquet
        df = pd.read_parquet(str(parquet_file)) 
        
        # 导出为 JSONL (orient='records', lines=True)，并确保非 ASCII 字符正确写入 (force_ascii=False)
        df.to_json(str(jsonl_file), orient='records', lines=True, force_ascii=False)
        
        logger.info(f"✅ Successfully converted {parquet_file.name} to {jsonl_file.name}")
        logger.info(f"Final file size: {jsonl_file.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to convert Parquet to JSONL: {e}")
        # 转换失败则抛出异常，阻止程序继续
        raise

def download_hotpotqa():
    """Download the test Parquet file for the HotpotQA dataset and convert it to JSONL."""
    
    try:
        logger.info(f"Downloading test Parquet file from {DATASET_REPO}/{PARQUET_FILENAME}...")
        
        # hf_hub_download 会将文件下载到本地缓存，并返回缓存的绝对路径（字符串）
        file_path_str = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=PARQUET_FILENAME,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )
        
        # 将返回的字符串路径转换为 Path 对象
        file_path = Path(file_path_str)
        logger.info(f"Downloaded Parquet file to temporary location: {file_path}")

        # 执行转换
        parquet_to_jsonl(file_path, HOTPOTQA_TEST_OUTPUT)
        
        logger.info(f"HotpotQA metadata saved to {HOTPOTQA_TEST_OUTPUT}")
        return HOTPOTQA_TEST_OUTPUT
    
    except Exception as e:
        # 捕获任何下载或转换过程中的异常
        logger.error(f"Failed to download and process HotpotQA test data: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download HotpotQA test dataset (hotpotqa/hotpot_qa)")
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
        logger.warning("No Hugging Face token provided. May fail if dataset is gated.")
        
    ensure_directories_exist()
    
    # 执行下载和转换
    download_hotpotqa()

    logger.info("✅ HotpotQA test dataset download and conversion completed!")

if __name__ == "__main__":
    main()