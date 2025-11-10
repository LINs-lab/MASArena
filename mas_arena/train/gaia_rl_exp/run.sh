#!/usr/bin/env bash
set -xeuo pipefail

# =================== Project Setup ===================
# Assumes run this script from the root of PROJECT_MEMORY_LAYER
PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# =================== Data/Model Setup ===================
DATA_ROOT=$PROJECT_ROOT/data
MODEL_PATH="Qwen/Qwen3-32B"

# Download and prepare data
if [ ! -f "$DATA_ROOT/gaia_validate.jsonl" ]; then
    echo "Downloading GAIA data..."
    python3 $PROJECT_ROOT/data/download/download_gaia.py
fi

# Convert data to Parquet if not already done
TRAIN_FILES=$PROJECT_ROOT/mas_arena/train/gaia_rl_exp/rl_data/train.parquet
if [ ! -f "$TRAIN_FILES" ]; then
    echo "Preparing GAIA data for RL..."
    python3 $PROJECT_ROOT/mas_arena/train/gaia_rl_exp/data_preparation.py \
        --data_path "$DATA_ROOT/gaia_validate.jsonl" \
        --output_dir "$PROJECT_ROOT/mas_arena/train/gaia_rl_exp/rl_data" \
        --split "train"
fi

# =================== Custom RL Config ===================
# Path to our custom agent loop and reward function
AGENT_LOOP_CONFIG_PATH=$PROJECT_ROOT/mas_arena/train/gaia_rl_exp/agent.yaml
REWARD_FN_FILE_PATH=$PROJECT_ROOT/mas_arena/train/gaia_rl_exp/reward_function.py
REWARD_FN_NAME="gaia_reward_func_with_memory"

# =================== RL Algorithm Params ===================
ADV_ESTIMATOR="grpo"
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
TRAIN_BATCH_SIZE=8  # 充分利用 8 块 GPU
MICRO_BATCH_SIZE=2  
ACTOR_LR=1e-6
GPUS_PER_NODE=8    
NNODES=1

# =================== Launch Training ===================
echo "Starting VERL training..."

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files="['$TRAIN_FILES']" \
    data.val_files="['$TRAIN_FILES']" \
    data.return_raw_chat=true \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.micro_batch_size=$MICRO_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.tensor_parallel_size=$GPUS_PER_NODE \
    actor_rollout_ref.model.quantization="int8" \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_LOOP_CONFIG_PATH \
    custom_reward_function.path="${REWARD_FN_FILE_PATH}" \
    custom_reward_function.name="${REWARD_FN_NAME}" \
    peft.lora.enable=true \
    peft.lora.r=64 \
    peft.lora.alpha=128 \
    peft.lora.dropout=0.05 \
    trainer.strategy="deepspeed_stage_2" \
    trainer.precision="bf16" \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.project_name="gaia_memory_rl" \
    trainer.experiment_name="qwen3-32b-grpo-rtx6000" \
    trainer.log_val_generations=10 \
    trainer.total_epochs=5