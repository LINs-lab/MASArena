import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import logging
import shutil
import pandas as pd

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GAIA-Downloader")

DATASET_REPO = "gaia-benchmark/GAIA"
# 这里的路径需对应 HF 仓库中的实际文件夹
REPO_PATHS = {
    "test": "2023/test",
    "validation": "2023/validation"
}

# 路径管理
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = SCRIPT_DIR.parent 
FILES_DIR = DATA_DIR / "files" / "gaia"

def ensure_directories_exist():
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    (FILES_DIR / "test").mkdir(parents=True, exist_ok=True)
    (FILES_DIR / "validate").mkdir(parents=True, exist_ok=True)

def get_metadata_path(split="test"):
    """获取 Parquet 格式的元数据路径"""
    repo_subpath = REPO_PATHS[split]
    # 根据你的发现，文件名现在是 metadata.parquet
    filename = f"{repo_subpath}/metadata.parquet"
    
    try:
        logger.info(f"正在从 Hugging Face 获取 {split} 版本的 Parquet 元数据...")
        local_cached_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN")
        )
        return Path(local_cached_path)
    except Exception as e:
        logger.error(f"获取 {split} 元数据失败 (404/401): {e}")
        return None

def download_files(metadata_path, split="test"):
    """使用 Pandas 解析 Parquet 并下载附件"""
    if not metadata_path or not metadata_path.exists():
        return

    # 确定本地保存目录
    split_dir_name = "test" if split == "test" else "validate"
    target_base_dir = FILES_DIR / split_dir_name
    repo_base_path = REPO_PATHS[split]

    # --- 解析 Parquet ---
    try:
        df = pd.read_parquet(metadata_path)
    except Exception as e:
        logger.error(f"解析 Parquet 文件失败: {e}")
        return

    logger.info(f"开始下载 {split} 附件，Parquet 中共有 {len(df)} 条记录")

    for _, entry in tqdm(df.iterrows(), total=len(df), desc=f"Downloading {split} files"):
        try:
            # 获取文件名列，GAIA 中通常叫 'file_name'
            file_name = entry.get("file_name")
            
            # 检查文件名是否有效 (排除空值或 NaN)
            if pd.isna(file_name) or not str(file_name).strip():
                continue

            file_name = str(file_name).strip()
            file_path_in_repo = f"{repo_base_path}/{file_name}"
            target_file_path = target_base_dir / file_name
            
            # 如果文件已存在则跳过
            if target_file_path.exists():
                continue

            target_file_path.parent.mkdir(parents=True, exist_ok=True)

            # 执行下载
            downloaded_temp_path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=file_path_in_repo,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN")
            )

            # 移动到目标位置
            shutil.copy2(downloaded_temp_path, target_file_path)

        except Exception as e:
            logger.debug(f"无法下载文件 {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="GAIA 附件下载器 (支持 Parquet 元数据)")
    parser.add_argument("--split", choices=["test", "validation", "both"], default="both")
    parser.add_argument("--token", help="Hugging Face Token")
    args = parser.parse_args()

    if args.token:
        os.environ["HF_TOKEN"] = args.token

    ensure_directories_exist()

    splits_to_process = ["test", "validation"] if args.split == "both" else [args.split]

    for s in splits_to_process:
        m_path = get_metadata_path(s)
        if m_path:
            download_files(m_path, s)

    logger.info("所有附件下载任务已完成！")

if __name__ == "__main__":
    main()