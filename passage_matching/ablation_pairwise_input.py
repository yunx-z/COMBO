import json
from copy import deepcopy
import random
import os

GPT_CTX_CNT = 10
DPR_CTX_CNT = 10
matching_method = "compatibility_2stage_optimal_unified"

for dataset in ['nq', 'tqa', 'webq']:
    for split in ['train', 'dev', 'test']:
        print(dataset, split)
        with open(f"../data/merge/{dataset}/sample_{GPT_CTX_CNT}gpt_{DPR_CTX_CNT}dpr/{matching_method}/{split}.json", 'r') as reader:
            items = json.load(reader)
        linearized_items = []
        shuffle_arm_order_items = []
        shuffle_psg_order_items = []
        for item in items:
            linearized_item = deepcopy(item)
            linearized_item['ctxs'] = []
            for ctx_pair in linearized_item['ctx_pairs']:
                linearized_item['ctxs'].append(ctx_pair[0])
                linearized_item['ctxs'].append(ctx_pair[1])
            del linearized_item['ctx_pairs']
            linearized_items.append(linearized_item)

            shuffle_arm_order_item = deepcopy(item)
            random.shuffle(shuffle_arm_order_item['ctx_pairs'])
            shuffle_arm_order_items.append(shuffle_arm_order_item)

            shuffle_psg_order_item = deepcopy(item)
            new_ctx_pairs = []
            for ctx_pair in shuffle_psg_order_item['ctx_pairs']:
                idx = [0, 1]
                random.shuffle(idx)
                new_ctx_pair = [ctx_pair[idx[0]], ctx_pair[idx[1]], ctx_pair[2]]
                new_ctx_pairs.append(new_ctx_pair)
            shuffle_psg_order_item['ctx_pairs'] = new_ctx_pairs
            shuffle_psg_order_items.append(shuffle_psg_order_item)

        with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_linearized.json", 'w') as writer:
            json.dump(linearized_items, writer, indent=4)
        ablation_items = {'shuffle_arm_order': shuffle_arm_order_items, 'shuffle_psg_order': shuffle_psg_order_items}
        for ablation in ablation_items:
            output_dir = f"../data/merge/{dataset}/sample_{GPT_CTX_CNT}gpt_{DPR_CTX_CNT}dpr/{matching_method}_{ablation}"
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{split}.json"), 'w') as writer:
                json.dump(ablation_items[ablation], writer, indent=4)

print("finish!")
