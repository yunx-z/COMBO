import json
import os
import copy
from tqdm import tqdm
from evaluation import has_answer
import random
from sentence_transformers import CrossEncoder

GPT_CTX_CNT = 10
DPR_CTX_CNT = 10



for dataset in ["webq"]:
    for split in ["train", "dev"]:
        print(dataset, split)
        infile = f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}.json"
        with open(infile, 'r') as reader:
            dataset_items = json.load(reader)
        k_dpr = dict()
        with open(f"checkpoint/infer_{dataset}_large_dpr_10_k_dpr_sample_{split}/final_output.txt", 'r') as reader:
            for line in reader:
                item = json.loads(line)
                if dataset == "wizard" and type(item['EM']) == type(True):
                    continue
                k_dpr[tuple(item['case_idx'])] = item['EM']
        for item in dataset_items:
            assert (item["idx"], -1, -1) in k_dpr
        k_1_dpr = dict()
        with open(f"checkpoint/infer_{dataset}_large_dpr_10_k_1_dpr_sample_{split}/final_output.txt", 'r') as reader:
            for line in reader:
                item = json.loads(line)
                if dataset == "wizard" and type(item['EM']) == type(True):
                    continue
                k_1_dpr[tuple(item['case_idx'])] = item['EM']
        for item in dataset_items:
            for j in range(DPR_CTX_CNT):
                if (item["idx"], -1, j) not in k_1_dpr:
                    print(f"{item['idx']} -1 {j} not in k_1_dpr")
        # continue
        k_1_dpr_1_gpt = dict()
        """
        with open(f"checkpoint/infer_{dataset}_large_dpr_10_k_1_dpr_1_gpt_sample_{split}/final_output.txt", 'r') as reader:
            for line in reader:
                item = json.loads(line)
                k_1_dpr_1_gpt[tuple(item['case_idx'])] = item['EM']
        """
        k_dpr_1_gpt = dict()
        """
        with open(f"checkpoint/infer_{dataset}_large_dpr_10_k_dpr_1_gpt_sample_{split}/final_output.txt", 'r') as reader:
            for line in reader:
                item = json.loads(line)
                k_dpr_1_gpt[tuple(item['case_idx'])] = item['EM']
        """
        positive = []
        neutral = []
        negative = []

        negative_I_cnt = 0
        negative_II_cnt = 0
        positive_negative_I_cnt = 0
        positive_negative_II_cnt = 0
        neutral_negative_I_cnt = 0
        neutral_negative_II_cnt = 0
        print("get negative pairs ...")
        for k_dpr_case_idx in tqdm(k_dpr):
            if dataset in ["nq", "tqa", "webq"]: 
                # idx start from 1
                dataset_item = dataset_items[k_dpr_case_idx[0]-1]
            else:
                # idx start from 0
                dataset_item = dataset_items[k_dpr_case_idx[0]]

            assert dataset_item['idx'] == k_dpr_case_idx[0]
            ctxs = dataset_item["ctxs"][:20]
            gpt_ctxs = ctxs[:10]
            dpr_ctxs = ctxs[10:20]
            """
            for ctx in ctxs:
                if ctx['title'] == "GPT Context":
                    gpt_ctxs.append(ctx)
                else:
                    dpr_ctxs.append(ctx)
            """
            assert len(gpt_ctxs) == GPT_CTX_CNT
            assert len(dpr_ctxs) == DPR_CTX_CNT
            question = dataset_item['question']
            if dataset in ['nq', 'tqa', 'webq'] and not question.endswith('?'):
                question += '?'
            answer = dataset_item['answers']
            for j in range(DPR_CTX_CNT):
                k_1_dpr_case_idx = (k_dpr_case_idx[0], -1, j)
                dpr_ctx = dpr_ctxs[j]
                if (dataset != "wizard" and not k_1_dpr[k_1_dpr_case_idx] and k_dpr[k_dpr_case_idx]) or (dataset == "wizard" and k_dpr[k_dpr_case_idx] - k_1_dpr[k_1_dpr_case_idx] > 0.1):
                    positive.append({"sentence1": f"{question}", "sentence2": f"({dpr_ctx['title']}) {dpr_ctx['text']}", "label": "positive", "type": "positive-I", "answer": answer})
                elif (dataset != "wizard" and k_1_dpr[k_1_dpr_case_idx] and not k_dpr[k_dpr_case_idx]) or (dataset == "wizard" and k_1_dpr[k_1_dpr_case_idx] - k_dpr[k_dpr_case_idx] > 0.1):
                    negative.append({"sentence1": f"{question}", "sentence2":  f"({dpr_ctx['title']}) {dpr_ctx['text']}", "label": "negative", "type": "negative-I", "answer": answer})
            """
            ems = [k_1_dpr[(k_dpr_case_idx[0], -1, j)] for j in range(DPR_CTX_CNT)]
            if ems.count(True) == 1:
                j = ems.index(True)
                dpr_ctx = dpr_ctxs[j]
                negative.append({"sentence1": f"{question}", "sentence2":  f"({dpr_ctx['title']}) {dpr_ctx['text']}", "label": "negative", "type": "negative-I", "answer": answer})
            elif ems.count(False) == 1:
                j = ems.index(False)
                dpr_ctx = dpr_ctxs[j]
                positive.append({"sentence1": f"{question}", "sentence2": f"({dpr_ctx['title']}) {dpr_ctx['text']}", "label": "positive", "type": "positive-I", "answer": answer})
            """

        print(f"{len(negative)} negative pairs in total: {negative_I_cnt} is Nagative I; {negative_II_cnt} is Negative II")
        print(f"{len(neutral)} neutral pairs in total: {neutral_negative_I_cnt} is Neutral-Nagative I; {neutral_negative_II_cnt} is Neutral Negative II")
        print(f"{len(positive)} all positive pairs in total: {positive_negative_I_cnt} is positive-from-Negative-I; {positive_negative_II_cnt} is positive-from-Negative-II")

        """
        if split == "train":
            # upsampling
            if len(negative) < len(positive):
                negative = negative * (len(positive) // len(negative)) + random.sample(negative, len(positive) % len(negative))
                assert len(positive) == len(negative)
        """
        outfile_path = f"../data/silver_evidentiality_2way_loose_sample/{dataset}/"
        os.makedirs(outfile_path, exist_ok=True)
        outfile = os.path.join(outfile_path, f"{split}.json")
        print(f"writing to {outfile} ..")
        with open(outfile, 'w') as writer:
            for pos in positive:
                writer.write(json.dumps(pos)+'\n')
            for neg in negative:
                writer.write(json.dumps(neg)+'\n')
            for neu in neutral:
                writer.write(json.dumps(neu)+'\n')

