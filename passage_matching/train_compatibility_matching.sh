#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=wangluxy1
#SBATCH --job-name=train_compatibility_matching_2way_sample_webq
#SBATCH --partition=spgpu
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=46GB
#SBATCH --time=07-00:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:


export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate memtf

python run_imbalanced_classification.py \
  --model_name_or_path checkpoint/silver_compatibility_2way_sample/nq_tqa/roberta-large \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --train_file ../data/silver_compatibility_2way_sample/webq/train.json \
  --validation_file ../data/silver_compatibility_2way_sample/webq/dev.json \
  --test_file ../data/silver_compatibility_2way_sample/webq/dev.json \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --num_train_epochs 7 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --seed 42 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --overwrite_output_dir \
  --output_dir checkpoint/silver_compatibility_2way_sample/webq/roberta-large \
