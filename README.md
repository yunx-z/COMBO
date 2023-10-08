# COMBO
Merging Generated and Retrieved Knowledge for Open-Domain QA

# Instructions 

## Setup 
The experiments in the paper were run using `python=3.7.0`, `torch==1.11.0` and `trasnformers==4.26.0`.

## Discriminator Training
To show you how to train the discriminators with silver labels, we will use NaturalQuestion to illustrate. 
### Step 1: Generate Silver Labels
```
cd passage_matching
python create_loo_data.py
bash infer_loo.sh
python create_evidentiality_data.py
python create_compatibility_data.py
```
### Step 2: Train and Infer with Discriminators
```
sbatch train_evidentiality.sh
sbatch train_compatibility.sh
bash submit_infer_compatibility_evidentiality_scores.sh
```
### Step 3: Create Passage Pairs
```
python merge_infer_scores.py
python build_psg_pairs.py --matching_method {random|compatibility_2stage_optimal}  --dataset {nq/tqa/webq}
```
where `compatibility_2stage_optimal` refers to COMBO method in paper.

## Reader Training
We modify codes from [atlas](https://github.com/facebookresearch/atlas). Please follow their instructions to set up environments properly.
```
cd ../training
sbatch scripts/train_simple_merge.sh   # for Direct Merging method
sbatch scripts/train_nq_webq_pair.sh   # for Random Matching / COMBO method
```
