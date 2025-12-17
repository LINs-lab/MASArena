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
DATASET_REPO = "openai/openai_humaneval"
TEST_METADATA_PATH = "openai_humaneval/test-00000-of-00001.parquet"  # 数据集在 HF repo 中的文件名

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# 假设脚本位于某个目录，而 'data' 目录位于该目录的父目录（或与脚本在同一级别，如果脚本在顶层）。
# 为了保持与 'data/humaneval_test.jsonl' 一致，我们需要确定 DATA_DIR。
# 如果 SCRIPT_DIR 就是项目根目录，那么 FILES_DIR 应该是 SCRIPT_DIR / "data"
# 如果您的脚本与第一个示例的路径结构相同，它假设 DATA_DIR 是 SCRIPT_DIR.parent
# 让我们明确定义输出目录为 'data'
DATA_DIR = SCRIPT_DIR.parent 
FILES_DIR = DATA_DIR  
HUMANEVAL_TEST_OUTPUT = FILES_DIR / "humaneval_test.jsonl" # 最终保存的 metadata 文件路径

def ensure_directories_exist():
    """确保必要的输出目录存在。"""
    # 创建目录（包括父目录），如果已存在则不报错
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {FILES_DIR}")

def convert_parquet_to_jsonl(parquet_path: Path, output_jsonl_path: Path):
    """读取 Parquet 文件，转换为 DataFrame，并保存为 JSONL 格式。"""
    try:
        import pandas as pd
    except ImportError:
        logger.error("Pandas is required to convert Parquet to JSONL. Please install it: pip install pandas pyarrow")
        return False
        
    logger.info(f"Converting {parquet_path.name} to JSONL...")
    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(parquet_path)
        
        # 保存为 JSONL 文件
        # orient='records', lines=True: 每行一个 JSON 对象
        # force_ascii=False: 允许非 ASCII 字符（如中文）
        df.to_json(output_jsonl_path, orient='records', lines=True, force_ascii=False)
        
        logger.info(f"Successfully converted and saved to {output_jsonl_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to convert Parquet to JSONL: {e}")
        return False


def download_metadata():
    """Download the test metadata file for the humaneval dataset."""
    metadata_path = TEST_METADATA_PATH
    output_file = HUMANEVAL_TEST_OUTPUT

    try:
        logger.info(f"Downloading test metadata from {DATASET_REPO}/{metadata_path}...")
        
        # 使用 hf_hub_download 下载 Parquet 文件
        # download_metadata 不应该直接处理 jsonl 而是下载原始文件
        file_path_str = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=metadata_path,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            # 指定本地缓存路径，以便后续处理
            cache_dir=FILES_DIR / ".cache"
        )
        file_path = Path(file_path_str)

        logger.info(f"Downloaded metadata to temporary/cached location: {file_path}")

        # 调用转换函数
        if convert_parquet_to_jsonl(file_path, output_file):
            # 可选：如果确定文件是文本格式且不大，可以记录大小
            # 这是一个 Parquet 文件，直接读取大小意义不大，但可以记录输出文件大小
            output_size = os.path.getsize(output_file)
            logger.info(f"Final JSONL file size: {output_size / 1024:.2f} KB")
            return output_file
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed to download test metadata: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download humaneval dataset (openai/openai_humaneval)")
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
    download_metadata()

    logger.info("✅ HumanEval test dataset download and conversion completed!")

if __name__ == "__main__":
    main()