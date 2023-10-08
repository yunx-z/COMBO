import json
import argparse
from tqdm import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from scipy.special import softmax

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str)
parser.add_argument("--split", type=str)
parser.add_argument("--part", type=int, default=0)
args = parser.parse_args()
dataset = args.dataset
split = args.split
part = args.part
PART_NUM = 8 if split == 'train' else 1
assert part in list(range(PART_NUM))

# compatibility_model = CrossEncoder(f'checkpoint/silver_compatibility_2way_sample/nq_tqa/roberta-large/', max_length=512)
compatibility_model = CrossEncoder('roberta-large', num_labels=2)
# compatibility_model = CrossEncoder(f'checkpoint/silver_compatibility_2way_sample/{dataset}/deberta-v3-large/', max_length=1024)
# compatibility_model = SentenceTransformer('all-MiniLM-L6-v2')
# compatibility_model = SentenceTransformer(f'checkpoint/silver_compatibility_2way_sample/{dataset}/all-MiniLM-L6-v2/')
compatibility_label_mapping = ['negative', 'positive']

def get_compatibility_score(sent_pairs):
    # return [{'label':'positive', 'score':0} for i in range(len(sent_pairs))]
    scores = compatibility_model.predict(sent_pairs, batch_size=128)
    scores = softmax(scores, axis=1)
    labels = [{'label':compatibility_label_mapping[score_max_id], 'score':max_score.item()} for score_max_id, max_score in zip(scores.argmax(axis=1), scores.max(axis=1))]
    return labels

"""
def get_biencoder_compatibility_score(sentences1, sentences2):
    embeddings1 = compatibility_model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = compatibility_model.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    labels = []
    for i in range(len(sentences1)):
        arow = []
        for j in range(len(sentences2)):
            score = cosine_scores[i][j].item()
            if score < 0:
                arow.append({'label': 'negative', 'score': score})
            else:
                arow.append({'label': 'positive', 'score': score})
        labels.append(arow)
    return labels
"""


print(dataset, split)
# with open(f"../data/analysis/{dataset}_sample_20gpt_100dpr_{split}.json", 'r') as reader:
# with open(f"../data/analysis/nq_sample_20gpt_50dpr_{split}_evidentiality_2way.json", 'r') as reader:
with open(f"../data/analysis/nq_sample_10gpt_10dpr_{split}_evidentiality_2way.json", 'r') as reader:
    items = json.load(reader)
part_len = len(items) // PART_NUM + 1
start_idx = part * part_len
end_idx = start_idx + part_len
items = items[start_idx:end_idx]

for item in tqdm(items):
    gpt_ctxs = item['ctxs'][:10]
    dpr_ctxs = item['ctxs'][10:20]
    """
    for ctx in item['ctxs']:
        if ctx['title'] == "GPT Context":
            gpt_ctxs.append(ctx)
        else:
            dpr_ctxs.append(ctx)
    """
    # assert len(gpt_ctxs) == len(dpr_ctxs)
    question = item['question']
    if dataset in ['nq', 'tqa', 'webq'] and not question.endswith('?'):
        question += '?'
    # compatibility_scores = get_biencoder_compatibility_score([c1['text'] for c1 in gpt_ctxs], [f"{question} ({c2['title']}) {c2['text']}" for c2 in dpr_ctxs])
    # compatibility_scores = [get_compatibility_score([(f"{question} ({c2['title']}) {c2['text']}", c1['text']) for c2 in dpr_ctxs]) for c1 in gpt_ctxs]
    # all_compatibility_scores = get_compatibility_score([(f"{question} ({c2['title']}) {c2['text']}", c1['text']) for c1 in gpt_ctxs for c2 in dpr_ctxs])
    pairs_to_infer = []
    for i, c1 in enumerate(gpt_ctxs):
        for j, c2 in enumerate(dpr_ctxs):
            if item["evidentiality_scores"][j]['label'] == "positive":
                pairs_to_infer.append((f"{question} ({c2['title']}) {c2['text']}", c1['text']))
    if len(pairs_to_infer) > 0:
        compatibility_scores_for_positive = get_compatibility_score(pairs_to_infer)
    else:
        compatibility_scores_for_positive = []
    all_compatibility_scores = []
    idx_pos = 0
    for i, c1 in enumerate(gpt_ctxs):
        for j, c2 in enumerate(dpr_ctxs):
            if item["evidentiality_scores"][j]['label'] == "positive":
                all_compatibility_scores.append(compatibility_scores_for_positive[idx_pos])
                idx_pos += 1
            else:
                all_compatibility_scores.append({'label' : 'negative', 'score' : 0})
    assert idx_pos == len(compatibility_scores_for_positive)
    compatibility_scores = []
    for i in range(0, len(all_compatibility_scores), len(dpr_ctxs)):
        compatibility_scores.append(all_compatibility_scores[i:i+len(dpr_ctxs)])
    assert len(compatibility_scores) == len(gpt_ctxs)
    item["compatibility_scores"] = compatibility_scores

if split == "train":
    outfile = f"../data/analysis/{dataset}_sample_20gpt_50dpr_{split}_compatibility_2way_part{part}.json"
else:
    outfile = f"../data/analysis/{dataset}_sample_20gpt_50dpr_{split}_compatibility_2way.json"
print("check out", outfile)
with open(outfile, 'w') as writer:
    json.dump(items, writer, indent=4)

"""
if split == "train":
    train_items = []
    for p in range(PART_NUM):
        with open(f"../data/analysis/{dataset}_sample_20gpt_100dpr_{split}_compatibility_2way_part{part}.json", 'r') as reader:
            items = json.load(reader)
            train_items.extend(items)
    with open(f"../data/analysis/{dataset}_sample_20gpt_100dpr_{split}_compatibility_2way.json", 'w') as writer:
        json.dump(train_items, writer, indent=4)
"""
