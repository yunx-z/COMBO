echo "#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=wangluxy1
#SBATCH --job-name=build_psg_pairs_${2}_${1}
#SBATCH --partition=largemem
#SBATCH --cpus-per-task=1
#SBATCH --mem=200GB
#SBATCH --time=01-00:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.log

# The application(s) to execute along with its input arguments and options:


export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate memtf

python build_psg_pairs.py --matching_method ${1}  --dataset ${2}" > slurm_.sh

sbatch slurm_.sh
