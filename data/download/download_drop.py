import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import logging
import pandas as pd
import numpy as np 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====== 修改这里：新数据集信息 (模仿 AIME25 脚本风格) ======
DATASET_REPO = "ucinlp/drop"
# 数据集在 HF repo 中的文件名 (DROP 的 validation split)
TEST_METADATA_PATH = "data/validation-00000-of-00001.parquet" 

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR.parent      # data 目录的绝对路径
FILES_DIR = DATA_DIR             # 保留 FILES_DIR 以保持风格一致
DROP_TEST_OUTPUT = DATA_DIR / "drop_test.jsonl" # 最终保存的 metadata 文件路径

def ensure_directories_exist():
    """Ensure the target directories for the output file exist."""
    # 模仿 AIME25 脚本，使用 FILES_DIR (此处与 DATA_DIR 相同)
    test_dir = FILES_DIR 
    # 创建目录（包括父目录），如果已存在则不报错 
    test_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {test_dir}")


def convert_ndarray_to_list(data):
    """Recursively converts numpy.ndarray objects within a dict/list to Python lists."""
    if isinstance(data, dict):
        return {k: convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(element) for element in data]
    elif isinstance(data, np.ndarray):
        # 核心转换：将 ndarray 转换为 Python list
        return data.tolist()
    else:
        return data


# 函数名修改为 download_metadata，更符合第一个脚本的风格
def download_metadata():
    """Download the DROP validation Parquet file, convert to JSONL, and save it as drop_test.jsonl."""
    metadata_path = TEST_METADATA_PATH
    output_file = DROP_TEST_OUTPUT
    split_name = "validation"

    # 1. 下载 Parquet 文件
    try:
        logger.info(f"Downloading {split_name} data from {DATASET_REPO}/{metadata_path}...")
        
        file_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=metadata_path,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )

        logger.info(f"Downloaded metadata to temporary location: {file_path}")

    except Exception as e:
        logger.error(f"Failed to download DROP {split_name} parquet file: {e}")
        return None

    # 2. 读取 Parquet 并转换为 JSONL
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Read {len(df)} entries from the parquet file.")

        # 转换为 JSONL 格式并保存
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {split_name} to JSONL"):
                
                row_dict = row.to_dict()
                
                # 转换 ndarray 为 list
                cleaned_data = convert_ndarray_to_list(row_dict)
                
                # dumps 为 JSONL 行
                json_line = json.dumps(cleaned_data, ensure_ascii=False)
                f_out.write(json_line + '\n')
        
        logger.info(f"Metadata saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to process or save DROP {split_name} data: {e}")
        return None # 返回 None 表示失败


def main():
    parser = argparse.ArgumentParser(description="Download DROP dataset validation split (as test)")
    # 模仿 AIME25 脚本，只保留 token 和 debug
    parser.add_argument("--token", help="Hugging Face token for accessing gated datasets")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 处理 token
    if args.token:
        os.environ["HF_TOKEN"] = args.token
        logger.info("Hugging Face token set")
    elif not os.environ.get("HF_TOKEN"):
        logger.warning("No Hugging Face token provided. You may encounter access issues.")

    # Ensure directories exist
    ensure_directories_exist()

    # 执行下载和转换
    download_metadata()

    logger.info("✅ DROP validation dataset download and conversion completed!")


if __name__ == "__main__":
    main()