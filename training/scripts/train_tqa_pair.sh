#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:
#SBATCH --account=wangluxy1
#SBATCH --job-name=tqa_sample_10gpt_10dpr_3b_pair_same_oracle_answer
#SBATCH --partition=spgpu
#SBATCH --cpus-per-task=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=46GB
#SBATCH --time=03-00:00:00
#SBATCH --export=ALL
#SBATCH --exclude=gl1522
#SBATCH --output=logs/%x-%j.log
##SBATCH --dependency=afterany:52425305

export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
#module load python3.9-anaconda/2021.11
#source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
#conda activate atlas-env


echo "Run started at:- "
date

DATA_DIR=./atlas_data
SIZE=3b
LR=5e-5
BATCH_SIZE=4
ACCU_STEPS=2
TEXT_MAX_LEN=400

port=$(shuf -i 15000-16000 -n 1)
TRAIN_FILE="${DATA_DIR}/merge/tqa/sample_10gpt_10dpr/same_oracle_answer/train.jsonl"
EVAL_FILES="${DATA_DIR}/merge/tqa/sample_10gpt_10dpr/same_oracle_answer/dev.jsonl"
TEST_FILES="${DATA_DIR}/merge/tqa/sample_10gpt_10dpr/same_oracle_answer/test.jsonl"
SAVE_DIR=${DATA_DIR}/experiments/
EXPERIMENT_NAME=tqa_sample_10gpt_10dpr_3b_pair_same_oracle_answer
TRAIN_STEPS=15000
EVAL_STEPS=1000
N_CTX=10

rm -rf ${SAVE_DIR}${EXPERIMENT_NAME}

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
    --name ${EXPERIMENT_NAME} \
    --checkpoint_dir ${SAVE_DIR} \
    --eval_freq ${EVAL_STEPS} \
    --log_freq 100 \
    --total_steps ${TRAIN_STEPS} \
    --warmup_steps 1000 \
    --save_freq ${EVAL_STEPS} \
    --main_port $port \
    --write_results \
    --task "qa" \
    --use_file_passages \

srun python evaluate.py \
    --use_gradient_checkpoint_reader \
    --name pred_${EXPERIMENT_NAME} \
    --generation_max_length 16 \
    --precision bf16 \
    --reader_model_type t5-${SIZE} \
    --text_maxlength ${TEXT_MAX_LEN} \
    --model_path ${SAVE_DIR}/${EXPERIMENT_NAME}/checkpoint/best_dev \
    --encoder_format "{query} {text}" \
    --eval_data ${TEST_FILES} \
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

