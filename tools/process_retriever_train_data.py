import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import datasets
import re
import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from jinja2 import Environment, BaseLoader

from datasets import load_dataset
from promptsource.templates import TemplateCollection
from data_utils.all2std import all2std
from data_utils.std2all import std2all
from data_utils.data_config import DATA_GROUP_CONFIG, DATA_CONFIG

from argparse import ArgumentParser
from arguments import add_data_args, add_runtime_args, add_retriever_args, add_rng_args

env = Environment(loader=BaseLoader)
env.filters["choice"] = random.choice


def get_convertable_matrix(all_data_names):
    m = defaultdict(defaultdict(int))
    # ni <- std <- nj
    for ni in all_data_names:
        for nj in all_data_names:
            if ni != nj and std2all[ni]["info"]["judge"](all2std[nj]["info"]):
                m[ni][nj] = 1

    return m


def get_convertable_adjlist(all_data_names):
    adjlist = {k:[] for k in all_data_names}
    # ni <- std <- nj
    for ni in all_data_names:
        for nj in all_data_names:
            if ni in std2all and nj in all2std:
                if ni != nj and std2all[ni]["info"]["judge"](all2std[nj]["info"]):
                    adjlist[ni].append(nj)

    return adjlist


def adjlist2matrix(adjlist, names):
    name_idx_map = {n:i for i,n in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=np.int32)
    for i, ni in enumerate(names):
        for nj in adjlist[ni]:
            matrix[i][name_idx_map[nj]] = 1
    
    return matrix


def clean_before_render(sample):
    return {k.replace("-", "_"): v for k, v in sample.items()}


def convert(data_name_s, data_name_t, sample_o):
    std_data = {"context": [], "question": [], "options": [], "label": [], "answer": []}
    for key_s, key_std in all2std[data_name_s]["rules"]:
        m = re.match(r"L\((.*)\)", key_s)
        if m is not None:
            std_data[key_std].extend(m.group(1).split(";"))
        else:
            rtemplate = env.from_string(key_s)
            sample_o = clean_before_render(sample_o)
            item = rtemplate.render(**sample_o)
            std_data[key_std].append(item)
        
        std_data["label"] = list(map(int, std_data["label"]))
        std_data["answer"].extend([std_data["options"][l] for l in std_data["label"] if l < len(std_data["options"])])
        std_data["label"].extend([std_data["options"].index(a) for a in std_data["answer"] if a in std_data["options"]])
        std_data["label"] = list(set(std_data["label"]))
        std_data["answer"] = list(set(std_data["answer"]))

    t_data = defaultdict(dict)
    for key_t, key_std in std2all[data_name_t]["rules"]:
        if isinstance(key_std, str):
            item = env.from_string(key_std).render(std_data)
        elif isinstance(key_std, list):
            item = [env.from_string(_ks).render(std_data) for _ks in key_std]
        else:
            raise ValueError()

        key_t_split = key_t.split(".")
        if len(key_t_split) == 1:
            t_data[key_t] = item
        else:
            t_data[key_t_split[0]][key_t_split[1]] = item

    if "label" in t_data and data_name_t not in ["social_i_qa_o"]:
        t_data["label"] = int(t_data["label"])
    
    if data_name_t == "yahoo_answers_topics_o":
        t_data["topic"] = int(t_data["topic"])
    
    if data_name_t == "circa_o":
        t_data["goldstandard1"] = int(t_data["goldstandard1"])
        t_data["goldstandard2"] = int(t_data["goldstandard2"])
    
    if data_name_t == "google_wellformed_query_o":
        t_data["rating"] = float(t_data["rating"])

    return t_data


def get_hard_neg_samples(args, data_name, p_name, template, convertable_data_names, all_data_names, all_data, num):
    if len(convertable_data_names) == 0:
        # if cannot construct hard samples, replace with easy samples
        return get_easy_neg_samples(args, data_name, all_data_names, all_data, num)

    neg_samples_str = []
    neg_labels = []
    trial = 0
    while len(neg_samples_str) < num:
        if trial > 2 * num:
            return get_easy_neg_samples(args, data_name, all_data_names, all_data, num)

        neg_data_name = random.choice(convertable_data_names)
        neg_data = all_data[neg_data_name]
        neg_sample = convert(neg_data_name, data_name, random.choice(neg_data["samples"][args.ret_source_split]))
        neg_sample = clean_before_render(neg_sample)
        try:
            applied_sample = template.apply(neg_sample)
        except:
            trial += 1
            continue

        if len(applied_sample) != 2 or not all(list(map(len, applied_sample))) > 0:
            trial += 1
            continue

        neg_sample_str = " ".join(applied_sample)
        
        if neg_sample_str not in neg_samples_str and check_str(neg_sample_str):
            neg_samples_str.append(neg_sample_str)
            neg_labels.append(f"{neg_data_name}-{p_name}")
        
        trial += 1
    
    return neg_samples_str, neg_labels


def get_easy_neg_samples(args, data_name, all_data_names, all_data, num):
    neg_samples_str = []
    neg_labels = []
    while len(neg_samples_str) < num:
        neg_data_name = data_name
        while neg_data_name == data_name:
            neg_data_name = random.choice(all_data_names)
        
        data = all_data[neg_data_name]
        p_name, template = random.choice(data["templates"])
        sample = random.choice(data["samples"][args.ret_source_split])
        sample = clean_before_render(sample)
        applied_sample = template.apply(sample)
        
        if len(applied_sample) != 2 or not all(list(map(len, applied_sample))) > 0:
            continue

        sample_str = " ".join(applied_sample)
        if sample_str not in neg_samples_str and check_str(sample_str):
            neg_samples_str.append(sample_str)
            neg_labels.append(f"{neg_data_name}-{p_name}")
            
    return neg_samples_str, neg_labels


def get_all_data(all_data_names):
    datasets.disable_caching()
    all_data = {}
    collection = TemplateCollection()
    for data_name in tqdm(all_data_names, desc="Getting Data"):
        print(data_name)
        config = DATA_CONFIG[data_name]
        data_files = {split: os.path.join(config.data_dir, f"{split}.jsonl") for split in ["train"]}
        data = load_dataset("json", data_files=data_files)
        main_name, sub_name = config.name
        print(main_name, sub_name)
        templates = collection.get_dataset(main_name, sub_name)
        
        all_data[data_name] = {
            "samples": data,
            "templates": [(p_name, templates[p_name]) for p_name in templates.all_template_names]                
        }
    
    return all_data


def check_str(s):
    if len(s.split(" ")) > 256:
        return False
    return True


def clean_str(args, s):
    s = re.sub(r"\n+", "\n", s)
    s = s.strip()
    return s


def parse_data_names(data_names):
    data_group_names = data_names.split("-")
    data_names = []
    for name in data_group_names:
        if name in DATA_GROUP_CONFIG:
            data_names.extend(DATA_GROUP_CONFIG[name])
        else:
            data_names.append(name)
    
    return data_names


def construct_data(args, all_data_names, all_data, hard_neg_convertable):
    
    max_num = args.ret_train_num_per_prompt + args.ret_eval_num_per_prompt
    
    for data_name in tqdm(all_data_names, desc="Constructing Data"):
        data = all_data[data_name]
        samples = data["samples"][args.ret_source_split].shuffle(seed=args.seed)
        templates = data["templates"]
        for p_name, template in templates:
            all_dpr_samples = []
            os.makedirs(os.path.join(args.save, data_name, p_name), exist_ok=True)
            num_samples, idx = 0, 0
            it = tqdm(range(max_num), desc=f"{data_name}-{p_name}").__iter__()
            while num_samples < max_num and idx < len(samples):
                # select pos_sample, neg_easy_sample, neg_hard_sample for each sample
                sample = samples[idx]
                sample = clean_before_render(sample)
                sample_str = " ".join(template.apply(sample))
                if not check_str(sample_str):
                    idx += 1
                    continue
                
                pos_sample_str = None
                trial = 0
                while True:
                    pos_idx = idx
                    while pos_idx == idx:
                        pos_idx = random.randint(0, len(samples) - 1)
                    pos_sample = samples[pos_idx]
                    pos_sample_str = " ".join(template.apply(pos_sample))
                    if check_str(pos_sample_str):
                        break
                    trial += 1
                    if trial > 2 * len(samples):
                        break
                if pos_sample_str is None:
                    print(data_name, p_name, "can't find suitable positive sample")
                    idx += 1
                    continue
                
                easy_neg_samples_str, easy_neg_labels = get_easy_neg_samples(args, data_name, all_data_names, all_data, args.ret_easy_neg_num)
                hard_neg_samples_str, hard_neg_labels = get_hard_neg_samples(args, data_name, p_name, template, hard_neg_convertable[data_name], all_data_names, all_data, args.ret_hard_neg_num)

                pos_sample_str = clean_str(args, pos_sample_str)
                easy_neg_samples_str = [clean_str(args, s) for s in easy_neg_samples_str]
                hard_neg_samples_str = [clean_str(args, s) for s in hard_neg_samples_str]
                
                dpr_data = {
                    "query": sample_str,
                    "pos_context": pos_sample_str,
                    "easy_neg_contexts": easy_neg_samples_str,
                    "easy_neg_labels": easy_neg_labels,
                    "hard_neg_contexts": hard_neg_samples_str,
                    "hard_neg_labels": hard_neg_labels,
                    "label": f"{data_name}-{p_name}",
                }
                
                all_dpr_samples.append(dpr_data)
                
                idx += 1
                num_samples += 1
                next(it)
            
            split_dpr_samples = {}
            split_dpr_samples["train"] = all_dpr_samples[:args.ret_train_num_per_prompt]
            split_dpr_samples["valid"] = all_dpr_samples[args.ret_train_num_per_prompt:]
            
            for split in ["train", "valid"]:
                with open(os.path.join(args.save, data_name, p_name, f"{split}.jsonl"), "w") as f:
                    for d in split_dpr_samples[split]:
                        f.write(json.dumps(d) + "\n")


def merge_data(args, all_data_names, all_data):
    os.makedirs(os.path.join(args.save, "merge"), exist_ok=True)
    for split in ["train", "valid"]:
        file_name = os.path.join(args.save, "merge", f"{split}.jsonl")
        f_merge = open(file_name, "w")
        for data_name in all_data_names:
            data = all_data[data_name]
            templates = data["templates"]
            for p_name, _ in templates:
                with open(os.path.join(args.save, data_name, p_name, f"{split}.jsonl")) as f:
                    for line in f:
                        f_merge.write(line)
        f_merge.close()
        
        os.system(f"shuf {file_name} -o {file_name}")


def get_arguments():
    parser = ArgumentParser()
    parser = add_rng_args(add_data_args(add_runtime_args(add_retriever_args(parser))))
    
    args = parser.parse_args()
    
    args.save = os.path.join(
        args.save,
        args.data_names,
        f"p{args.ret_pos_num}" +
        f"_en{args.ret_easy_neg_num}" +
        f"_hn{args.ret_hard_neg_num}" +
        f"_s{args.seed}"
    )
    
    return args


def main():

    args = get_arguments()
    
    os.makedirs(args.save, exist_ok=True)

    random.seed(args.seed)
    all_data_names = parse_data_names(args.data_names)

    hard_neg_convertable = get_convertable_adjlist(all_data_names)
    
    all_data = get_all_data(all_data_names)
    construct_data(args, all_data_names, all_data, hard_neg_convertable)

    merge_data(args, all_data_names, all_data)
    
    
if __name__ == "__main__":
    main()