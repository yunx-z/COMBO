import json
import os

for dataset in ["hotpotqa"]:
    for split in ["train", "dev", "test"]:
        """
        for gpt_cnt, dpr_cnt in [(10, 10)]:
            infile = f"../data/analysis/{dataset}_sample_{gpt_cnt}gpt_{dpr_cnt}dpr_{split}_linearized.json"
            if not os.path.exists(infile):
                continue
            print("infile", infile)
            with open(infile, 'r') as reader:
                items = json.load(reader)
            outfile = f"atlas_data/{dataset}_sample_{gpt_cnt}gpt_{dpr_cnt}dpr_{split}_linearized.jsonl"
            print("outfile", outfile)
            with open(outfile, 'w') as writer:
                for item in items:
                    item['passages'] = item.pop('ctxs')
                    for idx, psg in enumerate(item['passages']):
                        psg['id'] = str(idx)
                        psg['text'] = f"title: {psg['title']} text: {psg['text']}"
                    writer.write(json.dumps(item)+'\n')
        """
        """
        infile = f"../data/{dataset}/{split}.json"
        if not os.path.exists(infile):
            continue
        print("infile", infile)
        with open(infile, 'r') as reader:
            items = json.load(reader)
        outfile = f"atlas_data/{dataset}_dpr_{split}.jsonl"
        print("outfile", outfile)
        with open(outfile, 'w') as writer:
            for item in items:
                item['passages'] = item.pop('ctxs')
                for idx, psg in enumerate(item['passages']):
                    psg['id'] = str(idx)
                    psg['text'] = f"title: {psg['title']} text: {psg['text']}"
                writer.write(json.dumps(item)+'\n')
        """
        # for gpt_cnt, dpr_cnt in [(10, 10), (10, 20), (20, 20), (10, 50), (20, 20), (50, 50)]: 
        for gpt_cnt, dpr_cnt in [(10, 10)]: 
            # for method in ["compatibility_1stage_optimal", "compatibility_2stage", "compatibility_2stage_optimal", "compatibility_2stage_optimal_product_label", "compatibility_2stage_optimal_shuffle_arm_order", "compatibility_2stage_optimal_shuffle_psg_order", "random", "same_oracle_answer"]:
            for method in ["compatibility_2stage_optimal"]:
            # for method in ["compatibility_2stage_optimal_unified", "compatibility_2stage_optimal_unified_product_label", "compatibility_2stage_optimal_unified_shuffle_arm_order", "compatibility_2stage_optimal_unified_shuffle_psg_order", "compatibility_1stage_optimal_unified", "compatibility_2stage_unified"]:
                infile = f"../data/merge/{dataset}/sample_{gpt_cnt}gpt_{dpr_cnt}dpr/{method}/{split}.json"
                if not os.path.exists(infile):
                    continue
                print("infile", infile)
                with open(infile, 'r') as reader:
                    items = json.load(reader)
                outfile_path = f"atlas_data/merge/{dataset}/sample_{gpt_cnt}gpt_{dpr_cnt}dpr/{method}/"
                os.makedirs(outfile_path, exist_ok=True)
                outfile = os.path.join(outfile_path, f"{split}.jsonl")
                print("outfile", outfile)
                with open(outfile, 'w') as writer:
                    for item in items:
                        item['passages'] = []
                        for idx, ctx_pair in enumerate(item['ctx_pairs']):
                            c1, c2, score = ctx_pair[0], ctx_pair[1], ctx_pair[2]
                            psg = dict()
                            psg['id'] = str(idx)
                            psg['title'] = ""
                            psg['score'] = score
                            psg['gpt_has_answer'] = c1['has_answer']
                            psg['dpr_has_answer'] = c2['has_answer']
                            if dataset != 'hotpotqa':
                                psg['text'] = f"parametric knowledge: {c1['text']} retrieved knowledge: ({c2['title']}) {c2['text']}"
                            else:
                                psg['text'] = f"parametric knowledge: {c1['text']} retrieved knowledge: {c2['text']}"
                            item['passages'].append(psg)
                        item.pop('ctx_pairs')
                        writer.write(json.dumps(item)+'\n')
         
