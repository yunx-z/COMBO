#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --account=wangluxy0
#SBATCH --job-name=tqa_sample_10gpt_10dpr_large_pair_compatibility_2stage_optimal_unified_seed23
#SBATCH --partition=spgpu
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=46GB
#SBATCH --time=07-00:00:00
#SBATCH --export=ALL
#SBATCH --exclude=gl1522
#SBATCH --output=logs/%x-%j.log
##SBATCH --dependency=afterany:

export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
#module load python3.9-anaconda/2021.11
#source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
#conda activate atlas-env


echo "Run started at:- "
date

DATA_DIR=./atlas_data
SIZE=large
LR=5e-5
BATCH_SIZE=32
ACCU_STEPS=1
TEXT_MAX_LEN=400

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/merge/tqa/sample_10gpt_10dpr/compatibility_2stage_optimal_unified/train.jsonl"
EVAL_FILES="${DATA_DIR}/merge/tqa/sample_10gpt_10dpr/compatibility_2stage_optimal_unified/dev.jsonl"
TEST_FILES="${DATA_DIR}/merge/tqa/sample_10gpt_10dpr/compatibility_2stage_optimal_unified/test.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=tqa_sample_10gpt_10dpr_large_pair_compatibility_2stage_optimal_unified_lr${LR}
TRAIN_STEPS=10000
EVAL_STEPS=500
N_CTX=10
SEED=23

rm -rf ${SAVE_DIR}${EXPERIMENT_NAME}_seed${SEED}

srun python train.py \
    --shuffle \
    --use_gradient_checkpoint_reader \
    --precision bf16 \
    --shard_optim \
    --shard_grads \
    --target_maxlength 16 \
    --reader_model_type t5-${SIZE} \
    --dropout 0.1 \
    --weight_decay 0.01 \
    --lr ${LR} \
    --scheduler linear \
    --text_maxlength ${TEXT_MAX_LEN} \
    --model_path none \
    --encoder_format "{query} {text}" \
    --train_data ${TRAIN_FILE} \
    --eval_data ${EVAL_FILES} \
    --per_gpu_batch_size ${BATCH_SIZE} \
    --accumulation_steps ${ACCU_STEPS} \
    --n_context ${N_CTX} \
    --seed ${SEED} \
    --name ${EXPERIMENT_NAME}_seed${SEED} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq ${EVAL_STEPS} \
    --log_freq 100 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 500 \
    --save_freq ${EVAL_STEPS} \
    --main_port $port \
    --write_results \
    --task "qa" \
    --use_file_passages \

srun python evaluate.py \
    --use_gradient_checkpoint_reader \
    --name pred_${EXPERIMENT_NAME}_seed${SEED} \
    --generation_max_length 16 \
    --precision bf16 \
    --reader_model_type t5-${SIZE} \
    --text_maxlength ${TEXT_MAX_LEN} \
    --seed ${SEED} \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}_seed${SEED}/checkpoint/best_dev \
    --encoder_format "{query} {text}" \
    --eval_data ${TEST_FILES} \
    --per_gpu_batch_size ${BATCH_SIZE} \
    --n_context ${N_CTX} \
    --checkpoint_dir ${SAVE_DIR} \
    --main_port $port \
    --task "qa" \
    --use_file_passages \
    --write_results \

srun python evaluate.py \
    --use_gradient_checkpoint_reader \
    --name pred_${EXPERIMENT_NAME}_seed${SEED}_dev \
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


echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

