import random
from collections import defaultdict
import os
import re
from promptsource.templates import TemplateCollection
from datasets import load_dataset
import pickle
import multiprocessing
from functools import partial

from utils import save_rank, print_rank, get_rank, barrier
from torch.utils.data import Dataset

from .data_config import DATA_GROUP_CONFIG, DATA_CONFIG
from collections import defaultdict


class Encoder():
    def __init__(self):
        pass
    
    def initializer(self, arguments, func):
        Encoder.arguments = arguments
        Encoder.func = func
    
    def encode(self, dn, pn, sample):
        return Encoder.func(dn, pn, sample, *Encoder.arguments)
        

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=False):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.shot = shot
        self.pad_id = self.tokenizer.eos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.eos_token_id
        self.num = num
        self.as_pool = as_pool
        self.ratio = ratio
        self.gpt_max_length = args.gpt_max_length
        self.max_length = args.max_length
        self.max_length_per_sample = args.max_length_per_sample
        self.rng_sample = rng_sample
        self.rng_order = rng_order
        self.cur_epoch = 0
        self.delimiter_id = 198
        self.data_names = self._get_data_names(args.data_names)
        self.data_prompts, self.data_prompt_names = self._get_data_prompts(self.data_names)
        self.all_data = self._load_data(self.data_prompt_names)
        
        if self.as_pool and self.num > 0 and self.num == self.args.icl_pool_num:
            self.data = self._trunc_data(self.all_data)
        else:
            self.data = self.all_data

        self.label_sid_map = self._get_label_sid_map(self.data)
        self.option_sizes = self._get_option_sizes(self.data) if self.split != "train" else None

        for dn in self.data:
            for pn in self.data[dn]:
                log_str = f"{split} | {dn} | {pn} | data num: {len(self.data[dn][pn])}"
                print_rank(log_str)
                save_rank(log_str, os.path.join(args.save, "log.txt"))

    def _trunc_data(self, all_data):
        data = defaultdict(dict)
        for dn in all_data:
            for pn in all_data[dn]:
                data[dn][pn] = all_data[dn][pn][:self.num]
        return data        

    def _load_data(self, data_prompt_names):
        all_data = defaultdict(dict)

        # pool, encoder = None, None
            
        for dn, pn in data_prompt_names:
            if self.args.force_process or \
                DATA_CONFIG[dn].is_force_process(pn) or \
                    not os.path.exists(os.path.join(self._get_cache_path(dn, pn), "data.pkl")):
                if get_rank() == 0:
                    if self.args.data_process_workers > 0:
                        encoder = Encoder()
                        arguments = [self.args, self.split, self.tokenizer, self.data_prompts, 0, self.max_length_per_sample, False]
                        pool = multiprocessing.Pool(self.args.data_process_workers, initializer=encoder.initializer, initargs=(arguments, self._process_one_sample))
                        data = self._process_data_parallel(dn, pn, pool, encoder)                        
                    else:
                        data = self._process_data(dn, pn)
                    self._save_to_cache(dn, pn, data)
            barrier()

            data = self._load_from_cache(dn, pn)
            all_data[dn][pn] = data
        
        # if pool is not None:
        #     pool.close()

        return all_data
    
    def set_epoch(self, e):
        self.cur_epoch = e
    
    def _get_data_names(self, data_names):
        data_group_names = data_names.split("-")
        data_names = []
        for name in data_group_names:
            if name in DATA_GROUP_CONFIG:
                data_names.extend(DATA_GROUP_CONFIG[name])
            else:
                data_names.append(name)
        
        return data_names
    
    def _get_data_prompts(self, data_names):
        def _check_data_prompt(dn, pn):
            # overide
            train_prompts = self.args.train_prompts.split("-") if self.args.train_prompts is not None else None
            eval_prompts = self.args.eval_prompts.split("-") if self.args.eval_prompts is not None else None
            ck = DATA_CONFIG[dn].check_allowed(pn, self.split) and \
                    (self.split != "train" or train_prompts is None or pn.replace(" ", "_") in train_prompts) and \
                        (self.split == "train" or eval_prompts is None or pn.replace(" ", "_") in eval_prompts)
            return ck
        
        collection = TemplateCollection()
        data_prompts = {}
        data_prompt_names = []
        for data_name in data_names:
            prompts = collection.get_dataset(*DATA_CONFIG[data_name].name)
            data_prompts[data_name] = {}
            for prompt_name in prompts.all_template_names:
                if _check_data_prompt(data_name, prompt_name):
                    data_prompts[data_name][prompt_name] = prompts[prompt_name]
                    data_prompt_names.append((data_name, prompt_name))

        return data_prompts, data_prompt_names
    
    def _get_cache_path(self, data_name, prompt_name):
        data_dir = os.path.join(self.args.base_path, DATA_CONFIG[data_name].data_dir)
        num = self.args.icl_pool_num if self.as_pool else self.num
        prompt_name = prompt_name.replace("/", "-")
        prompt_type = "origin"
        end = ""
        bos = ""
        balance = "balance" if (self.split != "train" and self.args.balance_eval) else ""
        r2s = "r2s"
        trim = "trim"
        
        cache_path = os.path.join(data_dir, f"icl_new_cache/{self.split}/{self.ratio}/{num}/{balance}/{self.max_length_per_sample}/{self.args.seed_data}/{prompt_name}/{prompt_type}/{end}/{bos}/{r2s}/{trim}")
        return cache_path
    
    def get_all_cache_path(self):
        all_cache_pathes = defaultdict(dict)
        for dn, pn in self.data_prompt_names:
            all_cache_pathes[dn][pn] = self._get_cache_path(dn, pn)
        return all_cache_pathes

    def _load_from_cache(self, data_name, prompt_name):
        cache_path = self._get_cache_path(data_name, prompt_name)
        print_rank("load from", cache_path)

        data = None
        if os.path.exists(os.path.join(cache_path, "data.pkl")):
            with open(os.path.join(cache_path, "data.pkl"), "rb") as f:
                data = pickle.load(f)
        print_rank("load end")
        return data

    def _save_to_cache(self, data_name, prompt_name, data):
        cache_path = self._get_cache_path(data_name, prompt_name)
        print_rank("save to", cache_path)
        os.makedirs(cache_path, exist_ok=True)
        with open(os.path.join(cache_path, "data.pkl"), "wb") as f:
            pickle.dump(data, f)
        print_rank("save end")

    def _process_data(self, dn, pn):
        data_dir = os.path.join(self.args.base_path, DATA_CONFIG[dn].data_dir)
        dataset = load_dataset("json", data_files={self.split:os.path.join(data_dir, f"{self.split}.jsonl")})
        dataset_shuf = dataset[self.split].shuffle(seed=self.args.seed_data)
        num = self.args.icl_pool_num if self.as_pool else self.num
        
        if self.args.balance_eval and self.split != "train" and DATA_CONFIG[dn].option_id_space is not None:
            skip, sid = 0, 0
            data = []
            assert pn in DATA_CONFIG[dn].option_id_space or "ALL" in DATA_CONFIG[dn].option_id_space
            if pn in DATA_CONFIG[dn].option_id_space:
                option_id_space = DATA_CONFIG[dn].option_id_space[pn]
            else:
                option_id_space = DATA_CONFIG[dn].option_id_space["ALL"]
                
            max_num_per_option = num // len(option_id_space) if num >= 0 else -1
            sample_per_option = {oid: [] for oid in option_id_space}
            for lid, sample in enumerate(dataset_shuf):
                if lid % 500 == 0:
                    print_rank(f"{dn} | {self.split} | {pn} | lid: {lid} | sid: {sid} | skip: {skip}")
                out = self._process_one_sample(dn, pn, sample, self.args, self.split, self.tokenizer, self.data_prompts, lid, self.max_length_per_sample)
                if out is not None and out["option_label_id"] in sample_per_option:
                    if max_num_per_option < 0 or len(sample_per_option[out["option_label_id"]]) < max_num_per_option:
                        sample_per_option[out["option_label_id"]].append(out)
                    
            # trunc
            valid_num_per_option = min(map(len, sample_per_option.values()))
            sample_per_option = {i:x[:valid_num_per_option] for i,x in sample_per_option.items()}
            for x in sample_per_option.values():
                data.extend(x)
            
            for d in data:
                d["idxs"] = sid
                sid += 1
            
            assert num < 0 or sid <= num
            
        else:
            skip, sid = 0, 0
            data = []
            for lid, sample in enumerate(dataset_shuf):
                if lid % 500 == 0:
                    print_rank(f"{dn} | {self.split} | {pn} | lid: {lid} | sid: {sid} | skip: {skip}")
                out = self._process_one_sample(dn, pn, sample, self.args, self.split, self.tokenizer, self.data_prompts, lid, self.max_length_per_sample)
                if out is not None:
                    data.append({
                        "idxs": sid,
                        **out
                    })
                    sid += 1

                if num > 0 and sid >= num:
                    break

        return data

    def _process_data_parallel(self, dn, pn, pool:multiprocessing.Pool, encoder:Encoder):
        data_dir = os.path.join(self.args.base_path, DATA_CONFIG[dn].data_dir)
        dataset = load_dataset("json", data_files={self.split:os.path.join(data_dir, f"{self.split}.jsonl")})
        dataset_shuf = dataset[self.split].shuffle(seed=self.args.seed_data)
        print_rank("Shuf End")
        num = self.args.icl_pool_num if self.as_pool else self.num
        skip, sid = 0, 0
        data = []
        
        encoded_docs = pool.imap_unordered(partial(encoder.encode, dn, pn), dataset_shuf, 10)
        for lid, out in enumerate(encoded_docs):
            if out is not None:
                data.append({
                    "idxs": sid,
                    **out
                })
                sid += 1
            else:
                skip += 1
            
            if lid % 500 == 0:
                print_rank(f"{dn} | {self.split} | {pn} | lid: {lid} | sid: {sid} | skip: {skip}")
            
            if num > 0 and sid >= num:
                pool.terminate()
                pool.close()
                pool.join()
                break

        return data

    def _clean_before_render(self, sample):
        return {k.replace("-", "_"): v for k, v in sample.items()}

    def _process_one_sample(self, dn, pn, sample, args, split, tokenizer, data_prompts, lid, max_length_per_sample, verbose=True):
        prompt = data_prompts[dn][pn]
        sample = self._clean_before_render(sample)
        applied_sample = prompt.apply(sample)
        if len(applied_sample) != 2:
            if verbose:
                print_rank(f"Length of applied sample {len(applied_sample)} != 2")
            return None
        
        # str
        context_str, target_str = applied_sample
        options_str = None
        if split != "train" and DATA_CONFIG[dn].task_type == "cls":
            options_str = prompt.get_answer_choices_list(sample)
        
        context_str, target_str, options_str, _ = DATA_CONFIG[dn].get_post_fn(pn)((context_str, target_str, options_str, sample))
        
        if split != "train" and DATA_CONFIG[dn].task_type == "cls":
            if options_str is None:
                if verbose and lid == 0:
                    print_rank(f"Options is None in {split}, skip.", (dn, pn))
                return None
        
        context_str = context_str.strip()
        target_str = getattr(prompt.metadata, "concate_str", " ") + target_str.strip()

        context_str = re.sub("\n+", "\n", context_str)
        target_str = re.sub("\n+", "\n", target_str)
        context_str = re.sub("\s+", " ", context_str)
        target_str = re.sub("\s+", " ", target_str)
        
        if len(context_str) + len(target_str) > 5000:
            return None
        
        # tokenize
        context_ids = tokenizer.encode(context_str, add_special_tokens=False)
        target_ids = tokenizer.encode(target_str, add_special_tokens=False) + [self.delimiter_id]
                
        if len(context_ids) + len(target_ids) > max_length_per_sample:
            return None

        if len(target_ids) == 0:
            if verbose:
                print_rank("target ids length 0, skip.")
            raise ValueError()
        
        options_ids, option_label_id = None, None
        if split != "train" and DATA_CONFIG[dn].task_type == "cls":
            options_str = [getattr(prompt.metadata, "concate_str", " ") + x.strip() for x in options_str]
            options_str = [re.sub("\s+", " ", x) for x in options_str]
            
            options_ids = [tokenizer.encode(option_str, add_special_tokens=False) + [self.delimiter_id] for option_str in options_str]

            if len(context_ids) + max([len(x) for x in options_ids]) > max_length_per_sample:
                if verbose:
                    print_rank(dn + " " + dn + " " + "context + option too long, skip")
                return None

            if target_str not in options_str:
                if verbose:
                    print_rank(dn + " " + pn + " " + "Skip option bug sample " + str(target_str) + " " + str(options_str))
                if dn == "trec":
                    # bad fix...
                    return None
                else:
                    raise ValueError()
            
            option_label_id = options_str.index(target_str)

        return {
            "context_ids": context_ids,
            "target_ids": target_ids,
            "target_str": target_str,
            "options_str": options_str,
            "options_ids": options_ids,
            "option_label_id": option_label_id
        }

    def _get_label_sid_map(self, data):
        label_sid_map = defaultdict(partial(defaultdict, partial(defaultdict, list)))
        for dn in data:
            for pn in data[dn]:
                for samp in data[dn][pn]:
                    if DATA_CONFIG[dn].finit_label(pn):
                        if (dn == "poem_sentiment_o" and (samp["target_str"] in [" mixed\n", " mixed"])) \
                        or ("trec" in dn and (samp["target_str"] in [" Expression", " Expression\n"])) \
                        or ("cb" in dn and ((samp["target_str"] in [" Neither", " Neither\n"]) or (samp["target_str"] in [" Maybe", " Maybe\n"]))):
                            # bad fix...
                            continue
                        else:
                            label_sid_map[dn][pn][samp["target_str"]].append(samp["idxs"])
        
        return label_sid_map

    def _get_option_sizes(self, data):
        option_sizes = defaultdict(partial(defaultdict, list))
        for dn in data:
            if DATA_CONFIG[dn].task_type == "cls":
                for pn in data[dn]:
                    for samp in data[dn][pn]:
                        option_sizes[dn][pn].append(sum(map(len, samp["options_ids"])))
        return option_sizes

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()
