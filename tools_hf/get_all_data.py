import datasets
import json
import os
from datasets import load_dataset

datasets.disable_caching()

seed = 981217


# data_name = "commonsense_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "dream"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "quail"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "challenge"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "quartz"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "social_i_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "wiqa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "cosmos_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "qasc"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "quarel"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "sciq"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "wiki_hop"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "adversarial_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="adversarialQA")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "quoref"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "ropes"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "duorc"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="SelfRC")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/SelfRC".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/SelfRC/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "duorc"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="ParaphraseRC")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/ParaphraseRC".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/ParaphraseRC/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "hotpot_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="distractor")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/distractor/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/distractor/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "hotpot_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="fullwiki")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/fullwiki/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/fullwiki/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "wiki_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "amazon_polarity"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# test_num = dataset["test"].num_rows
# train_dataset = [x for x in dataset["train"]]
# d = {
#     "train": train_dataset[:-test_num],
#     "validation": train_dataset[-test_num:],
#     "test": dataset["test"],
# }
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in d[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "app_reviews"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# train_dataset = [x for x in dataset["train"]]
# d = {
#     "train": train_dataset[:-1000],
#     "validation": train_dataset[-1000:],
# }
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in d[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "imdb"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# test_num = 1000
# train_dataset = [x for x in dataset["train"]]
# d = {
#     "train": train_dataset[:-test_num],
#     "validation": train_dataset[-test_num:],
#     "test": dataset["test"],
#     "unsupervised": dataset["unsupervised"]
# }
# for split in ["train", "validation", "unsupervised", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in d[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "rotten_tomatoes"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "yelp_polarity"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# test_num = dataset["test"].num_rows
# train_dataset = [x for x in dataset["train"]]
# d = {
#     "train": train_dataset[:-test_num],
#     "validation": train_dataset[-test_num:],
#     "test": dataset["test"],
# }
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in d[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "ag_news"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# test_num = dataset["test"].num_rows
# train_dataset = [x for x in dataset["train"]]
# d = {
#     "train": train_dataset[:-test_num],
#     "validation": train_dataset[-test_num:],
#     "test": dataset["test"],
# }
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in d[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "dbpedia_14"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# test_num = dataset["test"].num_rows
# train_dataset = [x for x in dataset["train"]]
# d = {
#     "train": train_dataset[:-test_num],
#     "validation": train_dataset[-test_num:],
#     "test": dataset["test"],
# }
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in d[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "trec"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# test_num = dataset["test"].num_rows
# train_dataset = [x for x in dataset["train"]]
# d = {
#     "train": train_dataset[:-test_num],
#     "validation": train_dataset[-test_num:],
#     "test": dataset["test"],
# }
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in d[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "common_gen"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "wiki_bio"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "val", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
# os.system("cp /home/guyuxian/data_hf/wiki_bio/cache/val.jsonl /home/guyuxian/data_hf/wiki_bio/cache/validation.jsonl")


# data_name = "mrpc"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format("glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "paws"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="labeled_final")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/labeled_final/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/labeled_final/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "paws"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="labeled_swap")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/labeled_swap/".format(data_name), exist_ok=True)
# for split in ["train"]:
#     with open("/home/guyuxian/data_hf/{}/cache/labeled_swap/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "paws"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="unlabeled_final")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/unlabeled_final/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/unlabeled_final/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "qqp"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format("glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "copa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format("super_glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "hellaswag"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "story_cloze"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="2016")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/2016/".format(data_name), exist_ok=True)
# for split in ["validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/2016/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "story_cloze"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="2018")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/2018/".format(data_name), exist_ok=True)
# for split in ["validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/2018/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "anli"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# for r in [1, 2, 3]:
#     os.makedirs("/home/guyuxian/data_hf/{}/cache/r{}/".format(data_name, r), exist_ok=True)
#     for split, split_name in zip(["train_r{}".format(r), "dev_r{}".format(r), "test_r{}".format(r)], ["train", "validation", "test"]):
#         with open("/home/guyuxian/data_hf/{}/cache/r{}/{}.jsonl".format(data_name, r, split_name), "w") as f:
#             for x in dataset[split]:
#                 f.write(json.dumps(x) + "\n")


# data_name = "cb"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format("super_glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "rte"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format("super_glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "boolq"
# dataset = load_dataset(path="/home/lidong1/CodeRepo/icl_train/tools_hf/{}.py".format("super_glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/lidong1/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/lidong1/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "wsc"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format("super_glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "winogrande"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="winogrande_debiased")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/winogrande_debiased/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/winogrande_debiased/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "winogrande"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="winogrande_xl")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/winogrande_xl/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/winogrande_xl/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "wic"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format("super_glue"), name=data_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "xsum"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "samsum"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "multi_news"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "gigaword"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "cnn_dailymail"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="3.0.0")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/3.0.0".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/3.0.0/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "cnn_dailymail"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="1.0.0")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/1.0.0".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/1.0.0/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "cnn_dailymail"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="2.0.0")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/2.0.0".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/2.0.0/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "squad"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "art"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "circa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


# data_name = "circa_fix"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")

            
# data_name = "discovery"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="discovery")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            

# data_name = "emo"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "emotion"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "freebase_qa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "google_wellformed_query"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "liar"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "piqa"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            

# for sub_name in ['snli_format', 'tsv_format', 'dgem_format', 'predictor_format']:       
#     data_name = "scitail"
#     dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name=sub_name)
#     print(dataset)
#     dataset = dataset.shuffle(seed=seed)
#     os.makedirs("/home/guyuxian/data_hf/{}/cache/{}".format(data_name, sub_name), exist_ok=True)
#     for split in ["train", "validation", "test"]:
#         with open("/home/guyuxian/data_hf/{}/cache/{}/{}.jsonl".format(data_name, sub_name, split), "w") as f:
#             for x in dataset[split]:
#                 f.write(json.dumps(x) + "\n")
            
# for sub_name in ["full", "regular"]:
#     data_name = "swag"
#     dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name=sub_name)
#     print(dataset)
#     dataset = dataset.shuffle(seed=seed)
#     os.makedirs("/home/guyuxian/data_hf/{}/cache/{}".format(data_name, sub_name), exist_ok=True)
#     for split in ["train", "validation"]:
#         with open("/home/guyuxian/data_hf/{}/cache/{}/{}.jsonl".format(data_name, sub_name, split), "w") as f:
#             for x in dataset[split]:
#                 f.write(json.dumps(x) + "\n")
            
# sub_name = "tab_fact"       
# data_name = "tab_fact"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name=sub_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/{}".format(data_name, sub_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}/{}.jsonl".format(data_name, sub_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
   
   
# sub_name = "blind_test"       
# data_name = "tab_fact"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name=sub_name)
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/{}".format(data_name, sub_name), exist_ok=True)
# for split in ["test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}/{}.jsonl".format(data_name, sub_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "yahoo_answers_topics"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")

# for sub_name in ["ARC-Challenge", "ARC-Easy"]:
#     data_name = "ai2_arc"
#     dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name=sub_name)
#     print(dataset)
#     dataset = dataset.shuffle(seed=seed)
#     os.makedirs("/home/guyuxian/data_hf/{}/cache/{}".format(data_name, sub_name), exist_ok=True)
#     for split in ["train", "validation", "test"]:
#         with open("/home/guyuxian/data_hf/{}/cache/{}/{}.jsonl".format(data_name, sub_name, split), "w") as f:
#             for x in dataset[split]:
#                 f.write(json.dumps(x) + "\n")
            
            
# data_name = "climate_fever"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["test"]:
#     N = len(dataset[split])
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, "train"), "w") as f:
#         for i, x in enumerate(dataset[split]):
#             if i < int(0.8*N):
#                 f.write(json.dumps(x) + "\n")

#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, "validation"), "w") as f:
#         for i, x in enumerate(dataset[split]):
#             if i >= int(0.8*N):
#                 f.write(json.dumps(x) + "\n")
            
            
# data_name = "codah"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="codah")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/codah".format(data_name), exist_ok=True)
# for split in ["train"]:
#     with open("/home/guyuxian/data_hf/{}/cache/codah/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            

# data_name = "codah"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="fold_0")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/fold_0/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/fold_0/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")
            
            
# data_name = "financial_phrasebank"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name="sentences_allagree")
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/sentences_allagree".format(data_name), exist_ok=True)
# for split in ["train"]:
#     N = len(dataset[split])
#     with open("/home/guyuxian/data_hf/{}/cache/sentences_allagree/{}.jsonl".format(data_name, "train"), "w") as f:
#         for i, x in enumerate(dataset[split]):
#             if i < int(0.8*N):
#                 f.write(json.dumps(x) + "\n")

#     with open("/home/guyuxian/data_hf/{}/cache/sentences_allagree/{}.jsonl".format(data_name, "validation"), "w") as f:
#         for i, x in enumerate(dataset[split]):
#             if i >= int(0.8*N):
#                 f.write(json.dumps(x) + "\n")

  
# data_name = "medical_questions_pairs"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train"]:
#     N = len(dataset[split])
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, "train"), "w") as f:
#         for i, x in enumerate(dataset[split]):
#             if i < int(0.8*N):
#                 f.write(json.dumps(x) + "\n")

#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, "validation"), "w") as f:
#         for i, x in enumerate(dataset[split]):
#             if i >= int(0.8*N):
#                 f.write(json.dumps(x) + "\n")


# for sub_name in ["main", "additional"]:
#     data_name = "openbookqa"
#     dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name), name=sub_name)
#     print(dataset)
#     dataset = dataset.shuffle(seed=seed)
#     os.makedirs("/home/guyuxian/data_hf/{}/cache/{}".format(data_name, sub_name), exist_ok=True)
#     for split in ["train", "validation", "test"]:
#         with open("/home/guyuxian/data_hf/{}/cache/{}/{}.jsonl".format(data_name, sub_name, split), "w") as f:
#             for x in dataset[split]:
#                 f.write(json.dumps(x) + "\n")            


# data_name = "poem_sentiment"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")            


# data_name = "sick"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")            


# data_name = "yelp_full"
# dataset = load_dataset(path="/home/guyuxian/PPT/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/guyuxian/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "test"]:
#     with open("/home/guyuxian/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")            


# data_name = "mr"
# dataset = load_dataset(path="/home/lidong1/CodeRepo/icl_train/tools_hf/{}.py".format(data_name))
# print(dataset)
# dataset = dataset.shuffle(seed=seed)
# os.makedirs("/home/lidong1/data_hf/{}/cache/".format(data_name), exist_ok=True)
# for split in ["train", "validation", "test"]:
#     with open("/home/lidong1/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
#         for x in dataset[split]:
#             f.write(json.dumps(x) + "\n")


data_name = "roc_story"
dataset = load_dataset(path="/home/lidong1/CodeRepo/icl_train/tools_hf/{}.py".format(data_name))
print(dataset)
dataset = dataset.shuffle(seed=seed)
os.makedirs("/home/lidong1/data_hf/{}/cache/".format(data_name), exist_ok=True)
for split in ["train", "validation", "test"]:
    with open("/home/lidong1/data_hf/{}/cache/{}.jsonl".format(data_name, split), "w") as f:
        for x in dataset[split]:
            f.write(json.dumps(x) + "\n")