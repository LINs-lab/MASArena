import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== 修改这里：新数据集信息 ======
DATASET_REPO = "math-ai/aime25"
TEST_METADATA_PATH = "test.jsonl"  # 数据集在 HF repo 中的文件名

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR.parent      # data目录的绝对路径
FILES_DIR = DATA_DIR     
AIME_TEST_OUTPUT = DATA_DIR / "aime_test.jsonl" # 最终保存的 metadata 文件路径

def ensure_directories_exist():
    test_dir = FILES_DIR 
# 创建目录（包括父目录），如果已存在则不报错    
    test_dir.mkdir(parents=True, exist_ok=True)

def download_metadata():
    """Download the test metadata file for the AIME dataset."""
    metadata_path = TEST_METADATA_PATH
    output_file = AIME_TEST_OUTPUT

    try:
        logger.info(f"Downloading test metadata from {DATASET_REPO}/{metadata_path}...")
        
        file_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=metadata_path,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )

        logger.info(f"Downloaded metadata to temporary location: {file_path}")

        # 将文件内容复制到目标输出文件
        # 此处使用 'r'/'w' 模式而非 'rb'/'wb'，因为 jsonl 是文本文件
        with open(file_path, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            content = f_in.read()
            f_out.write(content)
            logger.info(f"Metadata content length: {len(content)} bytes")

        logger.info(f"Metadata saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Failed to download test metadata: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download AIME 2025 dataset (math-ai/aime25)")
    # 移除 --split 参数，因为只下载 test
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
    
    # 执行下载
    download_metadata()

    logger.info("✅ AIME25 test dataset download completed!")

if __name__ == "__main__":
    main()