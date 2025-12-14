import os
import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== 数据集信息 ======
DATASET_REPO = "Muennighoff/mbpp"
TEST_METADATA_FILENAME = "data/mbpp.jsonl" # 确认该文件本身就是 JSONL 格式

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# 假设 SCRIPT_DIR 是 project_root/scripts 或 project_root/download_script.py
# 假设 PROJECT_ROOT 是 project_root
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT

# 最终保存的 JSONL 文件路径 (原名为 IFEVAL_TEST_OUTPUT, 现更名为 MBPP_TEST_OUTPUT)
MBPP_TEST_OUTPUT = DATA_DIR / "mbpp_test.jsonl" 


def ensure_directories_exist():
    """确保必要的输出目录存在。"""
    # 创建 DATA_DIR (project_root)，如果已存在则不报错
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {DATA_DIR}")

    
def download_metadata():
    """下载 MBPP 数据集文件并将其移动到目标位置。"""
    
    try:
        logger.info(f"Downloading test metadata from {DATASET_REPO}/{TEST_METADATA_FILENAME}...")
        
        # 使用 hf_hub_download 下载 JSONL 文件
        file_path_str = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=TEST_METADATA_FILENAME,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            # 可选：指定缓存目录
            cache_dir=DATA_DIR / ".cache"
        )
        downloaded_file_path = Path(file_path_str)

        logger.info(f"Downloaded metadata to temporary/cached location: {downloaded_file_path}")

        # 1. 检查文件是否已存在，如果存在则删除旧的以确保复制成功
        if MBPP_TEST_OUTPUT.exists():
            MBPP_TEST_OUTPUT.unlink()
            logger.info(f"Removed existing file: {MBPP_TEST_OUTPUT}")
            
        # 2. 将下载的缓存文件复制/移动到最终目标位置
        # 由于 hf_hub_download 返回的是缓存路径，我们使用 shutil.copy2 来保留元数据
        shutil.copy2(downloaded_file_path, MBPP_TEST_OUTPUT)
        
        logger.info(f"Successfully copied JSONL file to {MBPP_TEST_OUTPUT}")
        
        output_size = os.path.getsize(MBPP_TEST_OUTPUT)
        logger.info(f"Final JSONL file size: {output_size / 1024:.2f} KB")
        
        # 可选：如果希望删除缓存文件（不建议，因为 Hugging Face Hub 会管理缓存）
        # os.remove(downloaded_file_path)
        # logger.info("Removed temporary downloaded file.")
        
        return MBPP_TEST_OUTPUT
            
    except Exception as e:
        logger.error(f"Failed to download test metadata: {e}")
        return None
    
def main():
    parser = argparse.ArgumentParser(description="Download MBPP dataset metadata")
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
    
    # 执行下载和复制
    download_metadata()

    logger.info("✅ MBPP test dataset download and copy completed!")

if __name__ == "__main__":
    main()