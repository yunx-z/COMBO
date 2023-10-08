#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=wangluxy1
#SBATCH --job-name=infer_hotpotqa_large_dpr_10_k_1_dpr_sample
#SBATCH --partition=spgpu
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=46GB
#SBATCH --time=07-00:00:00
#SBATCH --output=logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:


module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate fid

DATA_DIR=../atlas/atlas_data
SIZE=large
LR=1e-4
BATCH_SIZE=8
TEXT_MAX_LEN=400

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="../data/loo/k_1_dpr/hotpotqa/sample/train.jsonl"
EVAL_FILES="../data/loo/k_1_dpr/hotpotqa/sample/dev.jsonl"
TEST_FILES="../data/loo/k_1_dpr/hotpotqa/sample/test.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=hotpotqa_sample_10gpt_10dpr_large_simple_merge
TRAIN_STEPS=10000
EVAL_STEPS=1000
N_CTX=20
SEED=42


srun python evaluate.py \
    --use_gradient_checkpoint_reader \
    --name infer_hotpotqa_large_dpr_10_k_1_dpr_sample_dev \
    --generation_max_length 16 \
    --precision bf16 \
    --reader_model_type t5-${SIZE} \
    --text_maxlength ${TEXT_MAX_LEN} \
    --seed ${SEED} \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}_seed${SEED}/checkpoint/best_dev \
    --encoder_format "{query} {text}" \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size ${BATCH_SIZE} \
    --n_context ${N_CTX} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --task "qa" \
    --use_file_passages \
    --write_results \

srun python evaluate.py \
    --use_gradient_checkpoint_reader \
    --name infer_hotpotqa_large_dpr_10_k_1_dpr_sample_train \
    --generation_max_length 16 \
    --precision bf16 \
    --reader_model_type t5-${SIZE} \
    --text_maxlength ${TEXT_MAX_LEN} \
    --seed ${SEED} \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}_seed${SEED}/checkpoint/best_dev \
    --encoder_format "{query} {text}" \
    --eval_data ${TRAIN_FILES} \
    --per_gpu_batch_size ${BATCH_SIZE} \
    --n_context ${N_CTX} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --task "qa" \
    --use_file_passages \
    --write_results \

