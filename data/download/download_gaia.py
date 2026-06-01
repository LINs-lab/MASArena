import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import logging
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 固定使用官方源，避免 HF_ENDPOINT 镜像导致 404（GAIA 等仅在 huggingface.co 存在）
HF_ENDPOINT = "https://huggingface.co"

DATASET_REPO = "gaia-benchmark/GAIA"
# GAIA 自 2025-10 起使用 metadata.parquet，不再提供 metadata.jsonl
TEST_METADATA_PATH = "2023/test/metadata.parquet"
VALIDATION_METADATA_PATH = "2023/validation/metadata.parquet"

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR.parent  # data目录的绝对路径
FILES_DIR = DATA_DIR / "files"
GAIA_TEST_OUTPUT = DATA_DIR / "gaia_test.jsonl"
GAIA_VALIDATE_OUTPUT = DATA_DIR / "gaia_validate.jsonl"


def ensure_directories_exist():
    test_dir = FILES_DIR / "gaia" / "test"
    validate_dir = FILES_DIR / "gaia" / "validate"

    test_dir.mkdir(parents=True, exist_ok=True)
    validate_dir.mkdir(parents=True, exist_ok=True)


def parquet_to_jsonl(parquet_path: Path, output_jsonl_path: Path) -> bool:
    """将 Parquet 转为项目使用的 JSONL（每行一个 JSON 对象）。"""
    try:
        import pandas as pd
    except ImportError:
        logger.error("转换 Parquet 需要 pandas。请安装: pip install pandas pyarrow")
        return False
    try:
        df = pd.read_parquet(parquet_path)
        # 确保某些列存在，如果不存在则设为空
        if "file_name" not in df.columns:
            df["file_name"] = ""
            
        df.to_json(output_jsonl_path, orient="records", lines=True, force_ascii=False)
        logger.info(f"已转换并保存到 {output_jsonl_path}，共 {len(df)} 条")
        return True
    except Exception as e:
        logger.error(f"Parquet 转 JSONL 失败: {e}")
        return False


def download_metadata(split="test"):
    """Download metadata file for the specified split."""
    if split == "test":
        metadata_path = TEST_METADATA_PATH
        output_file = GAIA_TEST_OUTPUT
    else:
        metadata_path = VALIDATION_METADATA_PATH
        output_file = GAIA_VALIDATE_OUTPUT

    try:
        logger.info(f"正在从 {DATASET_REPO}/{metadata_path} 下载 {split} metadata...")
        # 显式指定 endpoint，覆盖环境变量
        file_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=metadata_path,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
            endpoint=HF_ENDPOINT,
        )
        logger.info(f"已下载到临时路径: {file_path}")

        # 转换为 jsonl
        if not parquet_to_jsonl(Path(file_path), output_file):
            return None
            
        return output_file
    except Exception as e:
        logger.error(f"下载 {split} metadata 失败: {e}")
        return None


def download_files(metadata_path, split="test"):
    """Download all files referenced in the metadata."""
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file {metadata_path} not found")
        return

    # 修改输出目录，对验证集使用正确的目录名
    if split == "test":
        output_dir = FILES_DIR / "gaia" / "test"
        repo_dir = "2023/test"
    else:
        output_dir = FILES_DIR / "gaia" / "validate"
        repo_dir = "2023/validation"

    logger.info(f"Files will be saved to: {output_dir}")

    # Read metadata file
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_lines = f.readlines()

    logger.info(f"Found {len(metadata_lines)} entries in {split} metadata")

    # 计数器
    files_found = 0
    files_downloaded = 0
    
    for line_idx, line in enumerate(tqdm(metadata_lines, desc=f"Processing {split} entries")):
        try:
            entry = json.loads(line)

            file_name = entry.get("file_name")
            # 确保 file_name 是字符串且不为空
            if isinstance(file_name, str):
                file_name = file_name.strip()
            else:
                file_name = ""

            if file_name:
                files_found += 1
                try:
                    # Determine file path in the repository
                    file_path_in_repo = f"{repo_dir}/{file_name}"
                    
                    # Download the file (显式指定 endpoint)
                    downloaded_path = hf_hub_download(
                        repo_id=DATASET_REPO,
                        filename=file_path_in_repo,
                        repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"),
                        endpoint=HF_ENDPOINT,
                    )

                    # Copy to our target directory
                    output_path = output_dir / file_name
                    shutil.copy2(downloaded_path, output_path)
                    files_downloaded += 1

                except Exception as e:
                    # 某些文件可能缺失，这在数据集中是可能的
                    # logger.warning(f"Failed to download file {file_name}: {e}")
                    pass
            elif line_idx < 5:
                # 仅调试用
                pass

        except json.JSONDecodeError:
            if line_idx < 5:
                logger.warning(f"Failed to parse JSON line in metadata: {line[:100]}...")
        except Exception as e:
            if line_idx < 5:
                logger.warning(f"Error processing metadata entry: {e}")

    logger.info(
        f"Completed downloading files for {split} split. Found {files_found} files, downloaded {files_downloaded} files.")


def main():
    parser = argparse.ArgumentParser(description="Download GAIA dataset")
    parser.add_argument("--split", choices=["test", "validation", "both"], default="both",
                        help="Which dataset split to download")
    parser.add_argument("--token", help="Hugging Face token for accessing gated datasets")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set token if provided
    if args.token:
        os.environ["HF_TOKEN"] = args.token
        logger.info("Hugging Face token set")
    else:
        logger.warning("No Hugging Face token provided. You may encounter access issues.")

    # Ensure directories exist
    ensure_directories_exist()

    # Download datasets based on split argument
    if args.split in ["test", "both"]:
        test_metadata_path = download_metadata("test")
        if test_metadata_path:
            download_files(test_metadata_path, "test")

    if args.split in ["validation", "both"]:
        validation_metadata_path = download_metadata("validation")
        if validation_metadata_path:
            download_files(validation_metadata_path, "validation")

    logger.info("Download process completed")


if __name__ == "__main__":
    main()
