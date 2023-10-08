#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=wangluxy1
#SBATCH --job-name=train_evidentiality_matching_2way_loose_sample_hotpotqa_deberta-v3-large
#SBATCH --partition=spgpu
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-gpu=46GB
#SBATCH --time=03-00:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:


export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate memtf

# CHANGE IMBALANCED WEIGHT!!!
python run_imbalanced_classification.py \
  --model_name_or_path microsoft/deberta-v3-large \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluation_strategy epoch \
  --train_file ../data/silver_evidentiality_2way_loose_sample/hotpotqa/train.json \
  --validation_file ../data/silver_evidentiality_2way_loose_sample/hotpotqa/dev.json \
  --test_file ../data/silver_evidentiality_2way_loose_sample/hotpotqa/dev.json \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 6e-6 \
  --num_train_epochs 4 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --seed 42 \
  --load_best_model_at_end \
  --metric_for_best_model f1 \
  --overwrite_output_dir \
  --output_dir checkpoint/silver_evidentiality_2way_loose_sample/hotpotqa/deberta-v3-large \
