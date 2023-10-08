#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=wangluxy1
#SBATCH --job-name=create_loo_data
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00-03:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:


export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate memtf

python -u create_loo_data.py
