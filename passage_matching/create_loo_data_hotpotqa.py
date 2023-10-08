import json
import os
import copy
from tqdm import tqdm

# for dataset in ["nq", "tqa", "fever", "wizard"]:
for dataset in ["hotpotqa"]:
    for split in ["dev", "train"]:
        print(dataset, split)
        infile = f"../data/analysis/{dataset}_sample_10gpt_50dpr_{split}.json"
        with open(infile, 'r') as reader:
            items = json.load(reader)
        k_1_dpr = []
        k_1_dpr_1_gpt = []
        k_dpr = []
        k_dpr_1_gpt = []
        for item in tqdm(items):
            ctxs = item["ctxs"][:20]
            gpt_ctxs = item["ctxs"][:10]
            dpr_ctxs = item["ctxs"][10:20]
            """
            for ctx in ctxs:
                if ctx['title'] == "GPT Context":
                    gpt_ctxs.append(ctx)
                else:
                    dpr_ctxs.append(ctx)
            """
            for j in range(len(dpr_ctxs)):
                k_1_dpr_item = copy.deepcopy(item)
                k_1_dpr_item["case_idx"] = [item["idx"], -1, j]
                k_1_dpr_item['ctxs'] = dpr_ctxs[:j] + dpr_ctxs[j+1:]
                k_1_dpr.append(k_1_dpr_item)
            
            """
            for j in range(len(dpr_ctxs)):
                for i in range(len(gpt_ctxs)):
                    k_1_dpr_1_gpt_item = copy.deepcopy(item)
                    k_1_dpr_1_gpt_item["case_idx"] = [item["idx"], i, j]
                    k_1_dpr_1_gpt_item['ctxs'] = dpr_ctxs[:j] + dpr_ctxs[j+1:] + [gpt_ctxs[i]]
                    k_1_dpr_1_gpt.append(k_1_dpr_1_gpt_item)
            """

            k_dpr_item = copy.deepcopy(item)
            k_dpr_item["case_idx"] = [item["idx"], -1, -1]
            k_dpr_item['ctxs'] = dpr_ctxs
            k_dpr.append(k_dpr_item)

            """
            for i in range(len(gpt_ctxs)):
                k_dpr_1_gpt_item = copy.deepcopy(item)
                k_dpr_1_gpt_item["case_idx"] = [item["idx"], i, -1]
                k_dpr_1_gpt_item['ctxs'] = dpr_ctxs + [gpt_ctxs[i]]
                k_dpr_1_gpt.append(k_dpr_1_gpt_item)
            """

        outfile_path = f"../data/loo/k_1_dpr/{dataset}/sample"
        os.makedirs(outfile_path, exist_ok=True)
        outfile = os.path.join(outfile_path, f"{split}.jsonl")
        print(f"writing to {outfile} ..")
        with open(outfile, 'w') as writer:
            for item in k_1_dpr:
                item['passages'] = item.pop('ctxs')
                for idx, psg in enumerate(item['passages']):
                    psg['id'] = str(idx)
                writer.write(json.dumps(item)+'\n')

        """
        outfile_path = f"../data/loo/k_1_dpr_1_gpt/{dataset}/sample"
        os.makedirs(outfile_path, exist_ok=True)
        outfile = os.path.join(outfile_path, f"{split}.jsonl")
        print(f"writing to {outfile} ..")
        with open(outfile, 'w') as writer:
            for item in k_1_dpr_1_gpt:
                item['passages'] = item.pop('ctxs')
                for idx, psg in enumerate(item['passages']):
                    psg['id'] = str(idx)
                writer.write(json.dumps(item)+'\n')
        """


        outfile_path = f"../data/loo/k_dpr/{dataset}/sample"
        os.makedirs(outfile_path, exist_ok=True)
        outfile = os.path.join(outfile_path, f"{split}.jsonl")
        print(f"writing to {outfile} ..")
        with open(outfile, 'w') as writer:
            for item in k_dpr:
                item['passages'] = item.pop('ctxs')
                for idx, psg in enumerate(item['passages']):
                    psg['id'] = str(idx)
                writer.write(json.dumps(item)+'\n')

        """
        outfile_path = f"../data/loo/k_dpr_1_gpt/{dataset}/sample"
        os.makedirs(outfile_path, exist_ok=True)
        outfile = os.path.join(outfile_path, f"{split}.jsonl")
        print(f"writing to {outfile} ..")
        with open(outfile, 'w') as writer:
            for item in k_dpr_1_gpt:
                item['passages'] = item.pop('ctxs')
                for idx, psg in enumerate(item['passages']):
                    psg['id'] = str(idx)
                writer.write(json.dumps(item)+'\n')
        """






