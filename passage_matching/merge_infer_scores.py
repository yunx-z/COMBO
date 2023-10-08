import json
from tqdm import tqdm

task2part = {'evidentiality': 4, 'compatibility':8}

for dataset in ['hotpotqa']:
    for split in ["train", "dev", "test"]:
        task2items = {'evidentiality': [], 'compatibility': []}
        for task in ['compatibility', 'evidentiality']:
        # for task in ['evidentiality']:
            print(dataset, split, task)
            if split == "train" and task == "evidentiality":
                for part in range(task2part[task]):
                    with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_{task}_2way_part{part}.json", 'r') as reader:
                        items = json.load(reader)
                        task2items[task].extend(items)
            else:
                with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_{task}_2way.json", 'r') as reader:
                    items = json.load(reader)
                    task2items[task].extend(items)
        # with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_{task}_2way.json", 'w') as writer:
        #     # json.dump(task2items['compatibility'], writer, indent=4)
        #     json.dump(task2items['evidentiality'], writer, indent=4)

        assert len(task2items['evidentiality']) == len(task2items['compatibility'])
        for evidentiality_item, compatibility_item in tqdm(zip(task2items['evidentiality'], task2items['compatibility']), total=len(task2items['evidentiality'])):
            assert evidentiality_item['idx'] == compatibility_item['idx']
            evidentiality_item['compatibility_scores'] = compatibility_item['compatibility_scores']
        """
        for compatibility_item in task2items['compatibility']:
            compatibility_item['evidentiality_scores'] = [{'label':"positive", 'score':1.0} for _ in range(50)]
        """
        with open(f"../data/analysis/{dataset}_sample_10gpt_10dpr_{split}_compatibility_2stage.json", 'w') as writer:
            # json.dump(task2items['compatibility'], writer, indent=4)
            json.dump(task2items['evidentiality'], writer, indent=4)

