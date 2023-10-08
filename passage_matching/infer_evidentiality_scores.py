import json
import argparse
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from scipy.special import softmax

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--part", type=int, default=0)
args = parser.parse_args()
dataset = args.dataset
split = args.split
part = args.part
PART_NUM = 4 if split == 'train' else 1
assert part in list(range(PART_NUM))


# evidentiality_model = CrossEncoder(f'checkpoint/silver_evidentiality_2way_loose_sample/{dataset}/deberta-v3-large')
evidentiality_model = CrossEncoder('roberta-large', num_labels=2)
# evidentiality_model = CrossEncoder(f'checkpoint/silver_evidentiality_2way_loose_sample/{dataset}/roberta-large')
# evidentiality_model = CrossEncoder(f'checkpoint/silver_evidentiality_2way_loose_sample/nq_tqa/roberta-large')
evidentiality_label_mapping = ['negative', 'positive']

def get_evidentiality_score(sent_pairs):
    scores = evidentiality_model.predict(sent_pairs, batch_size=50)
    scores = softmax(scores, axis=1)
    labels = [{'label':evidentiality_label_mapping[score_max_id], 'score':max_score.item()} for score_max_id, max_score in zip(scores.argmax(axis=1), scores.max(axis=1))]
    return labels


print(dataset, split)
with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}.json", 'r') as reader:
    items = json.load(reader)
print("total items", len(items))
part_len = len(items) // PART_NUM + 1
start_idx = part * part_len
end_idx = start_idx + part_len
items = items[start_idx:end_idx]

for item in tqdm(items):
    """
    gpt_ctxs = []
    dpr_ctxs = []
    for ctx in item['ctxs']:
        if ctx['title'] == "GPT Context":
            gpt_ctxs.append(ctx)
        else:
            dpr_ctxs.append(ctx)
    assert len(gpt_ctxs) == 20
    assert len(dpr_ctxs) == 100
    dpr_ctxs = dpr_ctxs[:50]
    # assert len(gpt_ctxs) == len(dpr_ctxs)
    """
    gpt_ctxs = item['ctxs'][:10]
    dpr_ctxs = item['ctxs'][10:20]
    question = item['question']
    if dataset in ['nq', 'tqa', 'webq'] and not question.endswith('?'):
        question += '?'
    if dataset != "hotpotqa":
        evidentiality_scores = get_evidentiality_score([(f"{question}", f"({c2['title']}) {c2['text']}") for c2 in dpr_ctxs])
    else:
        evidentiality_scores = get_evidentiality_score([(f"{question}", f"{c2['text']}") for c2 in dpr_ctxs])

    item["evidentiality_scores"] = evidentiality_scores

if split == "train":
    outfile = f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_evidentiality_2way_part{part}.json"
else:
    outfile = f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_evidentiality_2way.json"
print("check out", outfile)
with open(outfile, 'w') as writer:
    json.dump(items, writer, indent=4)

"""
if split == "train":
    train_items = []
    for p in range(PART_NUM):
        with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_evidentiality_2way_unified_part{p}.json", 'r') as reader:
            items = json.load(reader)
            train_items.extend(items)
    for idx, item in enumerate(train_items):
        if idx + 1 != item['idx']:
            print("not right", item['idx'])
    with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_evidentiality_2way_unified.json", 'w') as writer:
        json.dump(train_items, writer, indent=4)
    print("total train items w/ evi score", len(train_items))
"""
