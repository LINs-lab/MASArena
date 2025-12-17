import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd

folder = 'data/bbh'
# 只在文件夹不存在时才创建
import logging
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATASET_REPO = "lukaemon/bbh"
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

BBH_OUTPUT_DIR = SCRIPT_DIR.parent / "bbh"

BBH_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

def ensure_directories_exist():
    """确保 BBH_OUTPUT_DIR 存在."""
    if not BBH_OUTPUT_DIR.exists():
        BBH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"已创建文件夹: {BBH_OUTPUT_DIR.as_posix()}")
    else:
        logger.info(f"文件夹 {BBH_OUTPUT_DIR.as_posix()} 已存在，无需创建。")

def download_bbh():
    """迭代下载 BBH 的所有子任务并转换为 JSONL."""
    logger.info(f"开始从 {DATASET_REPO} 下载 BBH 子任务并转换为 JSONL...")
    
    tasks_succeeded = 0
    tasks_failed = 0

    for task_name in tqdm(BBH_SUBTASKS, desc="Processing BBH Subtasks"):
        parquet_path_in_repo = f"hf://datasets/{DATASET_REPO}/{task_name}/test-00000-of-00001.parquet"
        output_jsonl_path = BBH_OUTPUT_DIR / f"{task_name}.jsonl"

        if output_jsonl_path.exists():
            logger.warning(f"跳过任务 '{task_name}'：文件已存在于 {output_jsonl_path.name}")
            tasks_succeeded += 1
            continue

        try:
            logger.info(f"下载并转换任务: {task_name}")
            
            # 使用 pandas 读取 Parquet 文件
            # 注意：此处的 'hf://' 依赖于环境配置，例如 fsspec-HuggingFace
            df = pd.read_parquet(parquet_path_in_repo)
            
            # 写入 JSONL 文件
            df.to_json(
                output_jsonl_path.as_posix(), 
                orient='records', 
                lines=True, 
                force_ascii=False
            )
            
            logger.info(f"✅ 任务 '{task_name}' 成功保存到 {output_jsonl_path.as_posix()}")
            tasks_succeeded += 1
            
        except Exception as e:
            logger.error(f"❌ 处理任务 '{task_name}' 失败: {e}")
            tasks_failed += 1

    logger.info("--- Download Summary ---")
    logger.info(f"总任务数: {len(BBH_SUBTASKS)}")
    logger.info(f"成功任务数: {tasks_succeeded}")
    logger.info(f"失败任务数: {tasks_failed}")
    
def main():
    parser = argparse.ArgumentParser(description="Download BBH dataset (lukaemon/bbh) and convert to JSONL.")
    parser.add_argument("--token", help="Hugging Face token for accessing gated datasets.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    # Set token if provided
    if args.token:
        os.environ["HF_TOKEN"] = args.token
        logger.info("Hugging Face token set")
    else:
        logger.warning("未提供 Hugging Face token。如果数据集被门控，可能会失败。")
        
    # Ensure directories exist
    ensure_directories_exist()
    
    # Execute download
    download_bbh()
    
    logger.info("🎉 BBH 下载和转换过程完成!")

if __name__ == "__main__":
    main()