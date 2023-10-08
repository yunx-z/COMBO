for DATASET in hotpotqa
do
	for TASK in evidentiality
	do
		for SPLIT in dev test
		do
			echo "#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=wangluxy0
#SBATCH --job-name=infer_${TASK}_scores_${DATASET}_${SPLIT}
#SBATCH --partition=spgpu
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=02-00:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.log
##SBATCH --dependency=afterany:53779242

# The application(s) to execute along with its input arguments and options:


export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate memtf

python infer_${TASK}_scores.py --dataset ${DATASET} --part 0 --split ${SPLIT}" > slurm.sh
			sbatch slurm.sh
		done
		for PART in {0..7..1}
		do
			echo "#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --account=wangluxy0
#SBATCH --job-name=infer_${TASK}_scores_${DATASET}_train_part${PART}
#SBATCH --partition=spgpu
##SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=02-00:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x-%j.log
##SBATCH --dependency=afterany:53779242

# The application(s) to execute along with its input arguments and options:


export HF_HOME=/scratch/wangluxy_root/wangluxy0/yunxiang/cache/huggingface/
module load python3.9-anaconda/2021.11
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate memtf

python infer_${TASK}_scores.py --dataset ${DATASET} --part ${PART} --split train" > slurm.sh
			sbatch slurm.sh

		done
	done
done
