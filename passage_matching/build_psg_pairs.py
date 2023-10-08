import json
import random
import os
import argparse
from evaluation import f1_score
from scipy.special import softmax
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from hungarianalg.alg import hungarian
from copy import deepcopy
from collections import Counter
from pprint import pprint


# nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
# nli_label_mapping = ['contradiction', 'entailment', 'neutral']

DPR_CTX_CNT = 10
GPT_CTX_CNT = 10
total_gpt_cnt = None
total_dpr_cnt = None

DUMMY_CTX = {"id":-1, "title":"<pad>", "text":"<pad>", "has_answer":None}

def get_nli_score(sent_pairs):
    scores = nli_model.predict(sent_pairs)
    scores = softmax(scores, axis=1)
    labels = [{'label':nli_label_mapping[score_max_id], 'score':max_score.item()} for score_max_id, max_score in zip(scores.argmax(axis=1), scores.max(axis=1))]
    return labels

def resize_list(alist, new_size):
    len_alist = len(alist)
    newlist = [alist[i%len_alist] for i in range(new_size)]
    return newlist

def resize_matrix(matrix, new_row_cnt, new_col_cnt):
    original_row_cnt = len(matrix)
    original_col_cnt = len(matrix[0])
    # assert original_row_cnt >= new_row_cnt
    # assert original_col_cnt >= new_col_cnt
    new_matrix = []
    for i in range(new_row_cnt):
        new_row = []
        for j in range(new_col_cnt):
            new_row.append(matrix[i%original_row_cnt][j%original_col_cnt])
        new_matrix.append(new_row)
    return new_matrix

def create_view(compatibility_matrix, gpt_unselected_idx, dpr_unselected_idx):
    subset_view = []
    new2old_gpt_idx = dict()
    new2old_dpr_idx = dict()
    for c1_new_idx, c1_idx in enumerate(gpt_unselected_idx):
        arow = []
        new2old_gpt_idx[c1_new_idx] = c1_idx
        for c2_new_idx, c2_idx in enumerate(dpr_unselected_idx):
            new2old_dpr_idx[c2_new_idx] = c2_idx
            arow.append(compatibility_matrix[c1_idx][c2_idx])
        subset_view.append(arow)

    return subset_view, new2old_gpt_idx, new2old_dpr_idx

def pad_matrix(subset_view):
    padded_subset_view = []
    original_row_cnt = len(subset_view)
    original_col_cnt = len(subset_view[0])
    target_size = max(original_row_cnt, original_col_cnt)
    for i in range(target_size):
        new_row = []
        for j in range(target_size):
            if i >= original_row_cnt or j >= original_col_cnt:
                new_row.append(-1)
            else:
                new_row.append(subset_view[i][j])
        padded_subset_view.append(new_row)
    return padded_subset_view 

def enhance_compatibility_score(label):
    if label['score'] < 0.5 and label['label'] in ['contradiction', 'entailment']:
        label['label'] = 'neutral'
    return label

def get_max_label_idx(labels, target):
    max_value = -1
    max_value_idx = None
    for idx, l in enumerate(labels):
        if l['label'] == target and l['score'] > max_value:
            max_value = l['score']
            max_value_idx = idx
    return max_value_idx

def convert_2stage(item, option='hard_label'):
    for i in range(len(item['compatibility_scores'])):
        for j in range(len(item['compatibility_scores'][0])):
            if option == "product_label":
                if item['evidentiality_scores'][j]['label'] == 'negative':
                    item['evidentiality_scores'][j]['score'] = 1 - item['evidentiality_scores'][j]['score']
                if item['compatibility_scores'][i][j]['label'] == 'negative':
                    item['compatibility_scores'][i][j]['score'] = 1 - item['compatibility_scores'][i][j]['score'] 
                item['compatibility_scores'][i][j] = {'label': "N/A", 'score': item['evidentiality_scores'][j]['score'] * item['compatibility_scores'][i][j]['score']}
            else:
                if option == 'hard_label' and item['evidentiality_scores'][j]['label'] == "negative":
                    # DPR-irrelevant
                    item['compatibility_scores'][i][j] = {'label': "neutral", 'score': 0}
                else:
                    if item['compatibility_scores'][i][j]['label'] == "positive":
                        item['compatibility_scores'][i][j] = {'label': "entailment", 'score': item['compatibility_scores'][i][j]['score']}
                    else:
                        item['compatibility_scores'][i][j] = {'label': "contradiction", 'score': 1-item['compatibility_scores'][i][j]['score']}

def get_z_edges(all_compatible_pairs):
    z_edges = []
    for i, j, score in all_compatible_pairs:
        i_compatible = False
        j_compatible = False
        for ii, jj, _ in all_compatible_pairs:
            if i == ii and j == jj:
                continue
            elif i == ii:
                i_compatible = True
            elif j == jj:
                j_compatible = True
        if i_compatible and j_compatible:
            z_edges.append((i, j, score))
    return z_edges

def top_m_selection(compatibility_scores):
    results = []
    all_compatible_pairs = [(i, j, compatibility_scores[i][j]['score']) for i in range(len(compatibility_scores)) for j in range(len(compatibility_scores[0])) if compatibility_scores[i][j]['label'] == "entailment"]
    i_compatible_vertex = set([i for i, j, _ in all_compatible_pairs])
    j_compatible_vertex = set([j for i, j, _ in all_compatible_pairs])
    while len(all_compatible_pairs) > max(len(i_compatible_vertex), len(j_compatible_vertex)):
        z_edges = get_z_edges(all_compatible_pairs)
        # print("all_compatible_pairs", all_compatible_pairs)
        # print("z_edges", z_edges)
        if len(z_edges) == 0:
            break
        edge_to_remove = sorted(z_edges, key=lambda x : x[2], reverse=False)[0] # greedy remove, smallest compatible score
        #if len(z_edges) > 0:
        #    edge_to_remove = random.choice(z_edges)
        #else:
        #    edge_to_remove = random.choice(all_compatible_pairs)
        all_compatible_pairs.remove(edge_to_remove)

    results.extend(all_compatible_pairs)
    # if results == len(compatibility_scores):
    #    return results
    
    # print("compatible", results)
    all_confliciting_pairs = [(i, j) for i in range(len(compatibility_scores)) for j in range(len(compatibility_scores[0])) if compatibility_scores[i][j]['label'] == "contradiction" and i not in i_compatible_vertex and j not in j_compatible_vertex]
    i_conflicting_vertex = set([i for i, j in all_confliciting_pairs])
    j_conflicting_vertex = set([j for i, j in all_confliciting_pairs])
    for j in j_conflicting_vertex:
        results.append((-1, j, -1))
        # for ii, jj in all_confliciting_pairs:
        #    if jj == j:
        #        results.append((ii, jj))
        #        break
    #if len(results) >= len(compatibility_scores):
    #    return results[:len(compatibility_scores)]

    #print("compatible+conflicting", results)
    #print("i_compatible_vertex", i_compatible_vertex)
    #print("j_compatible_vertex", j_compatible_vertex)
    #print("i_conflicting_vertex", i_conflicting_vertex)
    #print("j_conflicting_vertex", j_conflicting_vertex)
    remaining_i_vertex = [i for i in range(len(compatibility_scores)) if i not in i_compatible_vertex and i not in i_conflicting_vertex]
    remaining_j_vertex = [j for j in range(len(compatibility_scores[0])) if j not in j_compatible_vertex and j not in j_conflicting_vertex]
    for i in remaining_i_vertex:
        results.append((i, -1, -2))
    #random.shuffle(remaining_i_vertex)
    #random.shuffle(remaining_j_vertex)
    #while len(results) < len(compatibility_scores):
    #    i = remaining_i_vertex.pop(0) if len(remaining_i_vertex) > 0 else -1
    #    j = remaining_j_vertex.pop(0) if len(remaining_j_vertex) > 0 else -1
    #    results.append((i, j))
    # if len(results) < 10:
    #    print("compatible+conflicting+dpr-irrelevant", results)
    #    exit(0)
    return results






def get_psg_pairs(item, matching_method):
    ctxs = item["ctxs"]
    gpt_ctxs = ctxs[:total_gpt_cnt]
    dpr_ctxs = ctxs[total_gpt_cnt:]
    """
    for ctx in ctxs:
        if ctx['title'] == "GPT Context":
            gpt_ctxs.append(ctx)
        else:
            dpr_ctxs.append(ctx)
    """
    # assert len(gpt_ctxs) == len(dpr_ctxs)
    gpt_ctxs = resize_list(gpt_ctxs, GPT_CTX_CNT)
    if GPT_CTX_CNT < DPR_CTX_CNT:
       gpt_ctxs = resize_list(gpt_ctxs, DPR_CTX_CNT)
    dpr_ctxs = resize_list(dpr_ctxs, DPR_CTX_CNT)
    psg_pairs = []
    if matching_method == 'random':
        random.shuffle(gpt_ctxs)
        random.shuffle(dpr_ctxs)
        psg_pairs = [(c1, c2, 0) for c1, c2 in zip(gpt_ctxs, dpr_ctxs)]
        random.shuffle(psg_pairs)
    elif matching_method == 'same_rank':
        psg_pairs = [(c1, c2, 0) for c1, c2 in zip(gpt_ctxs, dpr_ctxs)]
    elif matching_method == 'f1_unifiedqa_answer':
        # Is random shuffling necessary here?
        # random.shuffle(gpt_ctxs)
        # random.shuffle(dpr_ctxs)
        all_psg_pairs = [(c1, c2, f1_score(c1["unifiedqa_answer"], c2["unifiedqa_answer"])) for c1 in gpt_ctxs for c2 in dpr_ctxs]
        all_psg_pairs.sort(key=lambda x: x[2], reverse=True)
        gpt_ctxs_paired = [False for i in range(len(gpt_ctxs))]
        dpr_ctxs_paired = [False for i in range(len(dpr_ctxs))]
        for pair in all_psg_pairs:
            gpt_ctx_idx = gpt_ctxs.index(pair[0])
            dpr_ctx_idx = dpr_ctxs.index(pair[1])
            if not gpt_ctxs_paired[gpt_ctx_idx] and not dpr_ctxs_paired[dpr_ctx_idx]:
                psg_pairs.append(pair)
                gpt_ctxs_paired[gpt_ctx_idx] = True
                dpr_ctxs_paired[dpr_ctx_idx] = True
        assert len(psg_pairs) == len(gpt_ctxs) 
    elif matching_method == "compatibility_gpt_anchor_resort":
        gpt_ctxs_paired = [False for i in range(len(gpt_ctxs))]
        dpr_ctxs_paired = [False for i in range(len(dpr_ctxs))]

        compatibility_scores = item.pop("compatibility_scores")
        label_priority = ["positive", "negative"]
        for compatibility_label in label_priority:
            for c1_idx, c1 in enumerate(gpt_ctxs):
                for c2_idx, c2 in enumerate(dpr_ctxs):
                    if compatibility_scores[c1_idx][c2_idx]['label'] == compatibility_label and not gpt_ctxs_paired[c1_idx] and not dpr_ctxs_paired[c2_idx]:
                        score = compatibility_scores[c1_idx][c2_idx]['score']
                        c2['score'] = c2_idx
                        if compatibility_label == "negative":
                            score *= -1
                        psg_pairs.append((c1, c2, score))
                        gpt_ctxs_paired[c1_idx] = True
                        dpr_ctxs_paired[c2_idx] = True
        psg_pairs.sort(key=lambda x: x[0]['score'], reverse=True)
        assert len(psg_pairs) == len(gpt_ctxs)
    elif matching_method in ["compatibility_optimal", "compatibility_3way_optimal", "compatibility_3way_enhanced_optimal", "compatibility_3way_neutral_optimal", "compatibility_3way_neutral_enhanced_optimal", "compatibility_2stage_optimal_unified", "compatibility_2stage_optimal", "compatibility_2stage_optimal_degpt", "compatibility_2stage_optimal_topm_degpt", "compatibility_2stage_optimal_product_label", "compatibility_2stage_optimal_unified_product_label", "compatibility_1stage_optimal", "compatibility_1stage_optimal_unified"]:
        if "compatibility_2stage" in matching_method:
            if "product_label" in matching_method:
                convert_2stage(item, option='product_label')
            else:
                convert_2stage(item)
        elif "compatibility_1stage" in matching_method:
            convert_2stage(item, option='hard_label_1stage')
        compatibility_scores = item.pop("compatibility_scores")
        if "enhanced" in matching_method:
            compatibility_scores = [[enhance_compatibility_score(compatibility_scores[c1_idx][c2_idx]) for c2_idx in range(len(dpr_ctxs))] for c1_idx in range(len(gpt_ctxs))]
        label2weight = {"entailment":2, "contradiction":1, "neutral":0, "positive":2, "negative":1}
        if "compatibility_2stage" in matching_method or "compatibility_1stage" in matching_method:
            compatibility_matrix = [[compatibility_scores[c1_idx][c2_idx]['score'] for c2_idx in range(len(dpr_ctxs))] for c1_idx in range(len(gpt_ctxs))]
        else:
            compatibility_matrix = [[label2weight[compatibility_scores[c1_idx][c2_idx]['label']] for c2_idx in range(len(dpr_ctxs))] for c1_idx in range(len(gpt_ctxs))]
        if "topm" in matching_method:
            match_result = top_m_selection(compatibility_scores)
            # print("match_result", match_result)
        else:
            match_result = hungarian(compatibility_matrix).match
        for pair_idx in match_result:
            # print("debug pair_idx", pair_idx)
            if "topm" in matching_method:
                c1_idx, c2_idx, score = pair_idx
            else:
                c1_idx, c2_idx = pair_idx
                score = compatibility_matrix[c1_idx][c2_idx]
            c1 = deepcopy(gpt_ctxs[c1_idx]) if c1_idx != -1 else deepcopy(dpr_ctxs[c2_idx])
            c2 = deepcopy(dpr_ctxs[c2_idx]) if c2_idx != -1 else deepcopy(gpt_ctxs[c1_idx])
            # if "degpt" in matching_method and c1_idx != -1 and c2_idx != -1 and compatibility_scores[c1_idx][c2_idx]['label'] == "contradiction":
            #    c1['text'] = "<pad>"
            psg_pairs.append((c1, c2, score))
        psg_pairs.sort(key=lambda x: x[2], reverse=True)
        if matching_method in ["compatibility_2stage_optimal_topm_degpt"]:
            while len(psg_pairs) < 19:
                psg_pairs.append((DUMMY_CTX, DUMMY_CTX, -3))
        assert len(psg_pairs) == len(gpt_ctxs), f"len(psg_pairs):{len(psg_pairs)}, != len(gpt_ctxs):{len(gpt_ctxs)}"
    elif matching_method in ["compatibility_3way", "compatibility_3way_enhanced", "compatibility_3way_neutral", "compatibility_3way_neutral_enhanced", "compatibility_2stage", "compatibility_2stage_unified", "compatibility_1stage"]:
        gpt_ctxs_paired = [False for i in range(len(gpt_ctxs))]
        dpr_ctxs_paired = [False for i in range(len(dpr_ctxs))]
        if "compatibility_2stage" in matching_method:
            convert_2stage(item)
        elif "compatibility_1stage" in matching_method:
            convert_2stage(item, option='hard_label_1stage')

        compatibility_scores = item.pop("compatibility_scores")
        if "enhanced" in matching_method:
            compatibility_scores = [[enhance_compatibility_score(compatibility_scores[c1_idx][c2_idx]) for c2_idx in range(len(dpr_ctxs))] for c1_idx in range(len(gpt_ctxs))]
        label2score = {"entailment":1, "contradiction":3, "neutral":2}
        label_priority = ["entailment", "contradiction", "neutral"] 

        for compatibility_label in label_priority:
            for c1_idx, c1 in enumerate(gpt_ctxs):
                for c2_idx, c2 in enumerate(dpr_ctxs):
                    if compatibility_scores[c1_idx][c2_idx]['label'] == compatibility_label and not gpt_ctxs_paired[c1_idx] and not dpr_ctxs_paired[c2_idx]:
                        score = label2score[compatibility_label]
                        psg_pairs.append((c1, c2, score))
                        gpt_ctxs_paired[c1_idx] = True
                        dpr_ctxs_paired[c2_idx] = True
        assert len(psg_pairs) == len(gpt_ctxs)
    elif matching_method == 'compatibility_2stage_optimal_pad_selection' or matching_method in [f'compatibility_2stage_top_{k}_base_{b}_selection' for k in range(50) for b in range(10)]:
        convert_2stage(item)
        compatibility_scores = item.pop("compatibility_scores")
        if matching_method == 'compatibility_2stage_optimal_pad_selection':
            k = GPT_CTX_CNT
            k1 = 0
        else:
            k = int(matching_method.split('_')[-4])
            b = int(matching_method.split('_')[-2])
            gpt_compatible_idx = set()
            dpr_compatible_idx = set()
            all_compatible_psg_pairs = []
            for c2_idx, c2 in enumerate(dpr_ctxs):
                for c1_idx, c1 in enumerate(gpt_ctxs):
                    if compatibility_scores[c1_idx][c2_idx]['label'] in ["positive", "entailment"]:
                        all_compatible_psg_pairs.append([deepcopy(c1), deepcopy(c2), compatibility_scores[c1_idx][c2_idx]['score'], c1_idx, c2_idx])
                        gpt_compatible_idx.add(c1_idx)
                        dpr_compatible_idx.add(c2_idx)
            all_compatible_psg_pairs.sort(key=lambda x: x[2], reverse=True)
            k1 = min(max(len(gpt_compatible_idx), len(dpr_compatible_idx)), k//b)
            psg_pairs.extend(all_compatible_psg_pairs[:k1])
        gpt_selected_idx = set()
        dpr_selected_idx = set()
        for pair in psg_pairs:
            gpt_selected_idx.add(pair[3])
            dpr_selected_idx.add(pair[4])
        gpt_unselected_idx = [c1_idx for c1_idx, c1 in enumerate(gpt_ctxs) if c1_idx not in gpt_selected_idx]
        dpr_unselected_idx = [c2_idx for c2_idx, c2 in enumerate(dpr_ctxs) if c2_idx not in dpr_selected_idx]
        # random
        """
        random.shuffle(gpt_unselected_idx)
        random.shuffle(dpr_unselected_idx)
        for _ in range(k-k1):
            c1_idx = gpt_unselected_idx.pop()
            c2_idx = dpr_unselected_idx.pop()
            c1 = gpt_ctxs[c1_idx]
            c2 = dpr_ctxs[c2_idx]
            psg_pairs.append([deepcopy(c1), deepcopy(c2), compatibility_scores[c1_idx][c2_idx]['score'], c1_idx, c2_idx])
        """
        # optimal matching
        compatibility_matrix = [[compatibility_scores[c1_idx][c2_idx]['score'] for c2_idx in range(len(dpr_ctxs))] for c1_idx in range(len(gpt_ctxs))]
        subset_view, new2old_gpt_idx, new2old_dpr_idx = create_view(compatibility_matrix, gpt_unselected_idx, dpr_unselected_idx)
        padded_subset_view = pad_matrix(subset_view)
        match_result = hungarian(padded_subset_view).match
        all_incompatible_psg_pairs = []
        for pair_idx in match_result:
            c1_new_idx, c2_new_idx = pair_idx
            if c1_new_idx not in new2old_gpt_idx or c2_new_idx not in new2old_dpr_idx:
                continue
            c1_idx = new2old_gpt_idx[c1_new_idx]
            c2_idx = new2old_dpr_idx[c2_new_idx]
            score = compatibility_matrix[c1_idx][c2_idx]
            c1 = deepcopy(gpt_ctxs[c1_idx])
            c2 = deepcopy(dpr_ctxs[c2_idx])
            all_incompatible_psg_pairs.append((c1, c2, score, c1_idx, c2_idx))
        all_incompatible_psg_pairs.sort(key=lambda x: x[2], reverse=True)
        psg_pairs.extend(all_incompatible_psg_pairs[:k-k1])

        curr_psg_pairs_cnt = len(psg_pairs)
        for _ in range(k-curr_psg_pairs_cnt):
            psg_pairs.append([DUMMY_CTX, DUMMY_CTX, -1])

        if len(psg_pairs) != k:
            print("compatibility_matrix")
            pprint(compatibility_matrix)
            print("gpt_selected_idx", gpt_selected_idx)
            print("dpr_selected_idx", dpr_selected_idx)
            print("new2old_gpt_idx", new2old_gpt_idx)
            print("new2old_dpr_idx", new2old_dpr_idx)
            print("subset_view")
            pprint(subset_view)
            print("match_result", match_result)
            print("psg_pairs", psg_pairs)
            exit(0)

        psg_pairs = [[pair[0], pair[1], pair[2]] for pair in psg_pairs]
    elif matching_method in ['compatibility_top_10_all', 'compatibility_top_10_pos_selection', 'compatibility_top_10_pos_neg_selection', 'compatibility_3way_neutral_pos_neg_selection']:
        if "compatibility_2stage" in matching_method:
            convert_2stage(item)

        compatibility_scores = item.pop("compatibility_scores")
        all_pos_psg_pairs = [[deepcopy(c1), deepcopy(c2), compatibility_scores[c1_idx][c2_idx]['score']] for c2_idx, c2 in enumerate(dpr_ctxs) for c1_idx, c1 in enumerate(gpt_ctxs) if compatibility_scores[c1_idx][c2_idx]['label'] in ["positive", "entailment"]]
        all_neg_psg_pairs = [[deepcopy(c1), deepcopy(c2), -compatibility_scores[c1_idx][c2_idx]['score']] for c2_idx, c2 in enumerate(dpr_ctxs) for c1_idx, c1 in enumerate(gpt_ctxs) if compatibility_scores[c1_idx][c2_idx]['label'] in ["negative", "contradiction"] ]
        all_neutral_psg_pairs = [[deepcopy(c1), deepcopy(c2), compatibility_scores[c1_idx][c2_idx]['score']] for c2_idx, c2 in enumerate(dpr_ctxs) for c1_idx, c1 in enumerate(gpt_ctxs) if compatibility_scores[c1_idx][c2_idx]['label'] in ["neutral"]]
        # random.shuffle(all_pos_psg_pairs)
        # all_pos_psg_pairs.sort(key=lambda x: x[2], reverse=True)
        # random.shuffle(all_neg_psg_pairs)
        # all_neg_psg_pairs.sort(key=lambda x: x[2], reverse=True)
        random.shuffle(all_neutral_psg_pairs)

        all_psg_pairs = all_pos_psg_pairs + all_neg_psg_pairs + all_neutral_psg_pairs
        # random.shuffle(all_psg_pairs)
        # all_psg_pairs.sort(key=lambda x: x[2], reverse=True)
        if matching_method == 'compatibility_top_10_pos_selection':
            psg_pairs = all_pos_psg_pairs[:len(gpt_ctxs)]
            psg_pairs += all_neg_psg_pairs[:len(gpt_ctxs)-len(psg_pairs)]
        elif matching_method in ['compatibility_top_10_pos_neg_selection', 'compatibility_3way_neutral_pos_neg_selection']:
            psg_pairs = all_psg_pairs[:len(gpt_ctxs)]
        elif matching_method in ['compatibility_2stage_top_10']:
            psg_pairs = all_pos_psg_pairs[:len(gpt_ctxs)]
            psg_pairs += all_neg_psg_pairs[:len(gpt_ctxs)-len(psg_pairs)]
            gpt_ctxs_paired = [False for i in range(len(gpt_ctxs))]
            dpr_ctxs_paired = [False for i in range(len(dpr_ctxs))]
            for pair in psg_pairs:
                gpt_ctx_idx = gpt_ctxs.index(pair[0])
                dpr_ctx_idx = dpr_ctxs.index(pair[1])
                if pair[2] < 0: # contradiction pair, delete gpt passage
                    pair[0]['text'] = "<pad>"
                gpt_ctxs_paired[gpt_ctx_idx] = True
                dpr_ctxs_paired[dpr_ctx_idx] = True

            if len(psg_pairs) < len(gpt_ctxs):
                for pair in all_neutral_psg_pairs:
                    if len(psg_pairs) >= len(gpt_ctxs):
                        break
                    gpt_ctx_idx = gpt_ctxs.index(pair[0])
                    dpr_ctx_idx = dpr_ctxs.index(pair[1])
                    if not gpt_ctxs_paired[gpt_ctx_idx] and not dpr_ctxs_paired[dpr_ctx_idx]:
                        psg_pairs.append(pair)
                        gpt_ctxs_paired[gpt_ctx_idx] = True
                        dpr_ctxs_paired[dpr_ctx_idx] = True
        else:
            gpt_ctxs_paired = [False for i in range(len(gpt_ctxs))]
            dpr_ctxs_paired = [False for i in range(len(dpr_ctxs))]
            for pair in all_psg_pairs:
                gpt_ctx_idx = gpt_ctxs.index(pair[0])
                dpr_ctx_idx = dpr_ctxs.index(pair[1])
                if not gpt_ctxs_paired[gpt_ctx_idx] and not dpr_ctxs_paired[dpr_ctx_idx]:
                    psg_pairs.append(pair)
                    gpt_ctxs_paired[gpt_ctx_idx] = True
                    dpr_ctxs_paired[dpr_ctx_idx] = True
        assert len(psg_pairs) == len(gpt_ctxs) 
    elif matching_method in ['same_answer', 'same_unifiedqa_answer', 'same_oracle_answer']:
        random.shuffle(gpt_ctxs)
        random.shuffle(dpr_ctxs)
        gpt_ctxs_paired = [False for i in range(len(gpt_ctxs))]
        dpr_ctxs_paired = [False for i in range(len(dpr_ctxs))]
        for i, c1 in enumerate(gpt_ctxs):
            for j, c2 in enumerate(dpr_ctxs):
                if matching_method in ['same_answer', 'same_unifiedqa_answer']:
                    condition = c1['unifiedqa_answer'] == c2['unifiedqa_answer'] and c1['unifiedqa_answer'] != 'no answer>' 
                else:
                    # condition = c1['has_answer'] == c2['has_answer'] and c1['has_answer'] is not None 
                    condition = c1['has_answer'] is not None and c2['has_answer'] is not None
                if condition and not gpt_ctxs_paired[i] and not dpr_ctxs_paired[j]:
                    psg_pairs.append((c1, c2, 1))
                    gpt_ctxs_paired[i] = True
                    dpr_ctxs_paired[j] = True
        unpaired_gpt_ctxs = [gpt_ctxs[i] for i in range(len(gpt_ctxs_paired)) if not gpt_ctxs_paired[i]]
        unpaired_dpr_ctxs = [dpr_ctxs[i] for i in range(len(dpr_ctxs_paired)) if not dpr_ctxs_paired[i]]
        assert len(unpaired_gpt_ctxs) == len(unpaired_dpr_ctxs)
        for c1, c2 in zip(unpaired_gpt_ctxs, unpaired_dpr_ctxs):
            psg_pairs.append((c1, c2, 0))
        assert len(psg_pairs) == len(gpt_ctxs) 

        """
        relevant_gpt_ctxs = [ctx for ctx in gpt_ctxs if ctx['unifiedqa_answer'] != 'no answer>']
        relevant_dpr_ctxs = [ctx for ctx in dpr_ctxs if ctx['unifiedqa_answer'] != 'no answer>']
        if len(relevant_gpt_ctxs) == 0:
            relevant_gpt_ctxs = gpt_ctxs
            # print('*'*20, "no relevant_gpt_ctxs!", '*'*20)
        if len(relevant_dpr_ctxs) == 0:
            relevant_dpr_ctxs = dpr_ctxs
            # print('*'*20, "no relevant_dpr_ctxs!", '*'*20)
        for c1 in relevant_gpt_ctxs:
            for c2 in relevant_dpr_ctxs:
                if c1['unifiedqa_answer'] == c2['unifiedqa_answer'] and c1['unifiedqa_answer'] != 'no answer>':
                    psg_pairs.append((c1, c2, 0))
        diff = len(gpt_ctxs) - len(psg_pairs)
        random.shuffle(psg_pairs)
        if diff > 0:
            for i in range(diff):
                psg_pairs.append((random.choice(relevant_gpt_ctxs), random.choice(relevant_dpr_ctxs), 0))
        elif diff < 0:
            psg_pairs = psg_pairs[:len(gpt_ctxs)]
        assert len(psg_pairs) == len(gpt_ctxs)
        # for c1, c2, _ in psg_pairs:
        #    print(f"{c1['unifiedqa_answer']} - {c2['unifiedqa_answer']}")
        """
    elif matching_method in ["nli", "nli_enc"]:
        # Is random shuffling necessary here?
        # random.shuffle(gpt_ctxs)
        # random.shuffle(dpr_ctxs)

        gpt_ctxs_paired = [False for i in range(len(gpt_ctxs))]
        dpr_ctxs_paired = [False for i in range(len(dpr_ctxs))]
        nli_scores = item.pop("nli_scores")
        label2score = {"entailment":1, "contradiction":3, "neutral":2}
        label_priority = ["entailment", "neutral", "contradiction"] if matching_method == "nli_enc" else ["entailment", "contradiction", "neutral"] 
        for nli_label in label_priority:
            for c1_idx, c1 in enumerate(gpt_ctxs):
                for c2_idx, c2 in enumerate(dpr_ctxs):
                    if nli_scores[c1_idx][c2_idx]['label'] == nli_label and not gpt_ctxs_paired[c1_idx] and not dpr_ctxs_paired[c2_idx]:
                        score = label2score[nli_label]
                        psg_pairs.append((c1, c2, score))
                        gpt_ctxs_paired[c1_idx] = True
                        dpr_ctxs_paired[c2_idx] = True
        assert len(psg_pairs) == len(gpt_ctxs)
    elif matching_method in ["gpt_anchor_nli"]:
        for c1_idx, c1 in enumerate(gpt_ctxs):
            nli_scores = get_nli_score([(c2['text'], c1['text']) for c2 in dpr_ctxs])
            entailment_idx = get_max_label_idx(nli_scores, "entailment")
            contradiction_idx = get_max_label_idx(nli_scores, "contradiction")
            if entailment_idx is not None:
                psg_pairs.append((c1, dpr_ctxs[entailment_idx], 1))
            elif contradiction_idx is not None:
                psg_pairs.append((None, dpr_ctxs[contradiction_idx], 3))
            else:
                psg_pairs.append((c1, None, 2))
    elif matching_method in ["dpr_anchor_nli"]:
        for c2_idx, c2 in enumerate(dpr_ctxs):
            nli_scores = get_nli_score([(c2['text'], c1['text']) for c1 in gpt_ctxs])
            entailment_idx = get_max_label_idx(nli_scores, "entailment")
            if entailment_idx is not None:
                psg_pairs.append((gpt_ctxs[entailment_idx], c2, 1))
            else:
                psg_pairs.append((None, c2, 2.5))
    elif matching_method == 'debug':
        dpr_ctxs.reverse()
        psg_pairs = [(c1, c2, 0) for c1, c2 in zip(gpt_ctxs, dpr_ctxs)]
    else:
        raise ValueError("invalid matching method")
    return psg_pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--matching_method", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--part", type=int, default=0)
    args = parser.parse_args()
    matching_method = args.matching_method
    dataset = args.dataset
    part = args.part
    PART_NUM = 8 if args.split == 'train' else 1
    assert part in list(range(PART_NUM))
    splits = ["train", "dev", "test"] if args.split is None else [args.split]

    if dataset == "nq" and "unified" in matching_method:
        total_gpt_cnt = 20
        total_dpr_cnt = 50
    elif dataset == "webq" or "unified" in matching_method or "compatibility_1stage" in matching_method or "product_label" in matching_method:
        total_gpt_cnt = 10
        total_dpr_cnt = 10
    else:
        total_gpt_cnt = 10 if dataset == "hotpotqa" else 20
        total_dpr_cnt = 10 if dataset == "hotpotqa" else 100


    for split in splits:
        if matching_method == "same_answer":
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_unifiedqa_answer.json"
        elif matching_method in ["f1_unifiedqa_answer", "same_unifiedqa_answer"]:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_finetuned_unifiedqa_answer.json"
        elif "nli" in matching_method:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_nli.json"
        elif "unified" in matching_method:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_compatibility_2stage_unified.json"
        elif "compatibility_1stage" in matching_method:
            infile = f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_compatibility_2way.json"
        elif "compatibility_2stage" in matching_method:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_compatibility_2stage.json"
        elif "compatibility_3way_neutral" in matching_method:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_compatibility_3way_neutral.json"
        elif "compatibility_3way" in matching_method:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_compatibility_3way.json"
        elif "compatibility" in matching_method:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}_compatibility.json"
        else:
            infile = f"../data/analysis/{dataset}_sample_{total_gpt_cnt}gpt_{total_dpr_cnt}dpr_{split}.json"
        print(f"processing {infile} ...")
        with open(infile, 'r') as reader:
            items = json.load(reader)
        if args.split is not None:
            part_len = len(items) // PART_NUM + 1
            start_idx = part * part_len
            end_idx = start_idx + part_len
            items = items[start_idx:end_idx]

        psg_pair_cnts = []
        for item in tqdm(items):
            if "compatibility_2stage" in matching_method:
                item["evidentiality_scores"] = resize_list(item["evidentiality_scores"], DPR_CTX_CNT)
                item["compatibility_scores"] = resize_matrix(item["compatibility_scores"], GPT_CTX_CNT, DPR_CTX_CNT)
                if GPT_CTX_CNT < DPR_CTX_CNT:
                   item["compatibility_scores"] = resize_matrix(item["compatibility_scores"], DPR_CTX_CNT, DPR_CTX_CNT)
                # print(len(item["compatibility_scores"]), "row")
                # print(len(item["compatibility_scores"][0]), "col")

            psg_pairs = get_psg_pairs(item, matching_method)
            psg_pair_cnts.append(len(psg_pairs))
            item.pop("ctxs")
            item["ctx_pairs"] = psg_pairs
        print(Counter(psg_pair_cnts))

        outfile_path = f"../data/merge/{dataset}/sample_{GPT_CTX_CNT}gpt_{DPR_CTX_CNT}dpr/{matching_method}/"
        os.makedirs(outfile_path, exist_ok=True)
        if args.split is not None and split == "train":
            outfile = os.path.join(outfile_path, f"{split}_part{part}.json")
        else:
            outfile = os.path.join(outfile_path, f"{split}.json")
        print(f"writing to {outfile} ..")
        with open(outfile, 'w') as writer:
            json.dump(items, writer, indent=4)

