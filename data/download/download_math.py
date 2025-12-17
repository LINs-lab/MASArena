import os
import argparse
import shutil # 引入 shutil 用于文件复制
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
DATASET_REPO = "HuggingFaceH4/MATH-500"
TEST_METADATA_FILENAME = "test.jsonl" # 确认该文件本身就是 JSONL 格式

# 脚本路径和项目根目录的设置（模仿第一个脚本的风格）
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# 假设 SCRIPT_DIR 是 project_root/scripts，那么 DATA_DIR 是 project_root
PROJECT_ROOT = SCRIPT_DIR.parent 
DATA_DIR = PROJECT_ROOT
# 最终保存的 JSONL 文件路径 (原脚本硬编码为 "data/math_test.jsonl"，这里调整为项目根目录下的 math_test.jsonl，与第一个脚本的 AIME_TEST_OUTPUT 逻辑保持一致)
# 如果您希望最终文件在 DATA_DIR/data/math_test.jsonl，请将 DATA_DIR 改为 PROJECT_ROOT / "data"
MATH_TEST_OUTPUT = DATA_DIR / "math_test.jsonl" 


def ensure_directories_exist():
    """确保必要的输出目录存在。"""
    # 创建 DATA_DIR (project_root)，如果已存在则不报错
    # 如果您希望输出目录是 project_root/data，请确保 DATA_DIR 设置正确
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output directory exists: {DATA_DIR}")

    
def download_math():
    """下载 MATH-500 数据集文件并将其移动到目标位置。"""
    
    metadata_path = TEST_METADATA_FILENAME
    output_file = MATH_TEST_OUTPUT
    
    try:
        logger.info(f"Downloading test metadata from {DATASET_REPO}/{metadata_path}...")
        
        # 使用 hf_hub_download 下载 JSONL 文件
        file_path_str = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=metadata_path,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
        downloaded_file_path = Path(file_path_str)

        logger.info(f"Downloaded metadata to temporary/cached location: {downloaded_file_path}")

        # 将下载的缓存文件复制到最终目标位置
        # 由于文件是 JSONL 文本格式，我们可以直接复制
        
        # 1. 检查文件是否已存在，如果存在则删除旧的以确保复制成功
        if output_file.exists():
            output_file.unlink()
            logger.info(f"Removed existing file: {output_file}")
            
        # 2. 复制文件
        shutil.copy2(downloaded_file_path, output_file)
        
        logger.info(f"Successfully copied JSONL file to {output_file}")
        
        output_size = os.path.getsize(output_file)
        logger.info(f"Final JSONL file size: {output_size / 1024 / 1024:.2f} MB")
        
        return output_file
            
    except Exception as e:
        logger.error(f"Failed to download test metadata: {e}")
        return None
    
def main():
    parser = argparse.ArgumentParser(description="Download MATH-500 dataset (HuggingFaceH4/MATH-500)")
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
    download_math()

    logger.info("✅ MATH-500 test dataset download completed!")

if __name__ == "__main__":
    main()