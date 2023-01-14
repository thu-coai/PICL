import random
import os
import math
import pickle
import numpy as np
import torch
from copy import deepcopy

from icl_train.utils import print_args

if os.environ.get("CODE_BASE", "BMT") == "HF":
    from icl_train.utils import save_rank_hf as save_rank
    from icl_train.utils import print_rank_hf as print_rank
    from torch.distributed import get_rank, barrier
else:
    from icl_train.utils import save_rank_bmt as save_rank
    from icl_train.utils import print_rank_bmt as print_rank

from .icl_base_dataset import BaseDataset
from .data_config import DATA_CONFIG
from collections import Counter


class ICLDataset(BaseDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=True):
        super().__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool)
        if self.args.chunk_len + self.max_length_per_sample > self.gpt_max_length:
            self.chunk_len = self.gpt_max_length - self.max_length_per_sample
        else:
            self.chunk_len = self.args.chunk_len

        self.valid_max_length = {dn:{pn:self.max_length for pn in self.all_data[dn]} for dn in self.all_data}
        self.valid_max_length_per_sample = {dn:{pn:self.max_length_per_sample for pn in self.all_data[dn]} for dn in self.all_data}
        self.valid_max_length_all_demos = {dn:{pn:self.max_length_all_demos for pn in self.all_data[dn]} for dn in self.all_data}
        self.valid_max_length_all_context = {dn:{pn:self.max_length for pn in self.all_data[dn]} for dn in self.all_data} # for generation prefix
        self.demo_pool = None
    
    def _get_max_demo_length_rand(self, samp):
        raise NotImplementedError

    def _get_max_demo_length_share(self, pool_d):
        raise NotImplementedError

    def _get_chunked_demo_ids(self, all_demo_ids, chunk_len, all_demo_ids_only_target=None):
        chunked_demo_ids = [[]]
        chunked_demo_ids_only_target = [[]] if all_demo_ids_only_target is not None else None
        for i in range(len(all_demo_ids)):
            assert len(all_demo_ids[i]) <= chunk_len, all_demo_ids[i]
            _demo_ids = all_demo_ids[i][1:] if self.args.add_bos else all_demo_ids[i]
            if all_demo_ids_only_target is not None:
                _demo_ids_only_target = all_demo_ids_only_target[i][1:] if self.args.add_bos else all_demo_ids_only_target[i]
            if len(chunked_demo_ids[-1]) + len(_demo_ids) + 1 <= chunk_len:
                chunked_demo_ids[-1].extend(_demo_ids)
                if all_demo_ids_only_target is not None:
                    chunked_demo_ids_only_target[-1].extend(_demo_ids_only_target)
            else:
                chunked_demo_ids.append(_demo_ids)
                if all_demo_ids_only_target is not None:
                    chunked_demo_ids_only_target.append(_demo_ids_only_target)
        
        if len(chunked_demo_ids) > 1:
            chunked_demo_ids.pop(-1)
            chunked_demo_ids_only_target.pop(-1)
        
        if self.args.add_bos:
            chunked_demo_ids = [[self.bos_id] + x for x in chunked_demo_ids]
            if all_demo_ids_only_target is not None:
                chunked_demo_ids_only_target = [[-1] + x for x in chunked_demo_ids_only_target]
        
        return chunked_demo_ids, chunked_demo_ids_only_target

    def _from_demo_idx_to_ids(self, data_name, prompt_name, d):
        pool_d = self.demo_pool[data_name][prompt_name]
        demo_idxs = d["demo_idxs"]
        
        all_demo_ids = [pool_d[idx]["context_ids"] + pool_d[idx]["target_ids"] for idx in demo_idxs]
        all_demo_ids_only_target = [[-100] * len(pool_d[idx]["context_ids"]) + pool_d[idx]["target_ids"] for idx in demo_idxs]
        
        if self.chunk_len > 0:
            all_demo_ids, all_demo_ids_only_target = self._get_chunked_demo_ids(all_demo_ids, self.chunk_len, all_demo_ids_only_target)
        
        all_demo_ids = [np.array(x) for x in all_demo_ids]
        all_demo_ids_only_target = [np.array(x) for x in all_demo_ids_only_target]
        
        return {
            "all_demo_ids": all_demo_ids,
            "all_demo_ids_only_target": all_demo_ids_only_target,
        }

    def _get_demo_cache_path(self, type, shot, pool_cache):
        demo_cache = os.path.join(pool_cache, f"demo_cache/{self.split}/{type}/{shot}/{self.max_length_per_sample}/{self.max_length}_{self.max_length_all_demos}")
        return demo_cache

    def _get_demo_idxs_rand(self, samp, pool_d, pool_sid, shot, pool_split):
        demo_idxs = self.rng_sample.sample(pool_sid, shot+1)
        
        # if demos are from the same pool, the test sample should not appear in the demos
        if pool_split == self.split and samp["idxs"] in demo_idxs:
            demo_idxs.pop(demo_idxs.index(samp["idxs"]))
        else:
            demo_idxs.pop()
        assert pool_split != self.split or samp["idxs"] not in demo_idxs
        
        demo_lengths = [len(pool_d[idx]["context_ids"]) + len(pool_d[idx]["target_ids"]) for idx in demo_idxs]

        while sum(demo_lengths) > self._get_max_demo_length_rand(samp):
            demo_idxs.pop()
            demo_lengths.pop()
            
        return demo_idxs, demo_lengths

    def build_icl_demos_rand(self, demo_pool, pool_split, pool_caches=None):
        log_str = f"{self.split} Sample icl train data rand"
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        
        self.demo_pool = demo_pool # for __getitem__
        
        for data_name in self.data:
            for prompt_name in self.data[data_name]:
                d = self.data[data_name][prompt_name]
                pool_d = demo_pool[data_name][prompt_name]
                pool_sid = [sample["idxs"] for sample in pool_d]
                if len(d) == 0:
                    continue
                real_shots = []
                # shot
                shot = min(self.shot, len(pool_sid) - 1)
                tmp_shot = DATA_CONFIG[data_name].get_shot(prompt_name, self.split)
                if tmp_shot is not None:
                    shot = tmp_shot
                
                if self.args.rand_shot_num:
                    shot = self.rng_sample.randint(0, shot)

                if pool_caches is not None:
                    demo_cache_path = self._get_demo_cache_path("rand", shot, pool_caches[data_name][prompt_name])
                    if self.args.force_process_demo or \
                        DATA_CONFIG[data_name].is_force_process_demo(prompt_name) or \
                            not os.path.exists(os.path.join(demo_cache_path, "demo.pkl")):
                        print_rank("Process demo idx and save to", demo_cache_path)
                        if get_rank() == 0:
                            all_demo_info = [self._get_demo_idxs_rand(samp, pool_d, pool_sid, shot, pool_split) for samp in d]
                            os.makedirs(demo_cache_path, exist_ok=True)
                            with open(os.path.join(demo_cache_path, "demo.pkl"), "wb") as f:
                                pickle.dump(all_demo_info, f)
                    barrier()

                    print_rank("Load demo idxs from", demo_cache_path)
                    with open(os.path.join(demo_cache_path, "demo.pkl"), "rb") as f:
                        all_demo_info = pickle.load(f)

                valid_max_length = 0
                valid_max_length_per_sample = 0
                valid_max_length_all_demos = 0
                for sid, samp in enumerate(d):
                    if pool_caches is not None:
                        demo_idxs, demo_lengths = all_demo_info[sid]
                    else:
                        demo_idxs, demo_lengths = self._get_demo_idxs_rand(samp, pool_d, pool_sid, shot, pool_split)
                    
                    # double check the test sample should not appear in the demos
                    assert pool_split != self.split or samp["idxs"] not in [pool_d[idx]["idxs"] for idx in demo_idxs]
                    
                    real_shots.append(len(demo_idxs))
                    
                    valid_max_length = max(valid_max_length, sum(demo_lengths) + len(samp["context_ids"]) + len(samp["target_ids"]))
                    valid_max_length_per_sample = max(valid_max_length_per_sample, len(samp["context_ids"]) + len(samp["target_ids"]))
                    valid_max_length_all_demos = max(valid_max_length_all_demos, sum(demo_lengths))
                    
                    samp.update({
                        "demo_idxs": demo_idxs
                    })
                
                self.valid_max_length[data_name][prompt_name] = valid_max_length
                self.valid_max_length_all_demos[data_name][prompt_name] = valid_max_length_all_demos
                self.valid_max_length_per_sample[data_name][prompt_name] = valid_max_length_per_sample

                mean_real_shots = np.mean(real_shots) if len(real_shots) > 0 else 0
                log_str = f"{self.split} | Average real shots | {data_name} | {prompt_name} | {mean_real_shots}"
                print_rank(log_str)
                save_rank(log_str, os.path.join(self.args.save, "log.txt"))

    def _get_demo_ids_share(self, data_name, prompt_name, pool_d, pool_l, pool_sid, max_length_all_demos, balance):
        # shot
        shot = min(self.shot, len(pool_sid) - 1)
        tmp_shot = DATA_CONFIG[data_name].get_shot(prompt_name, self.split)
        if tmp_shot is not None:
            shot = tmp_shot
        
        origin_shot = shot
        length_all_demos = max_length_all_demos + 1
        repeat_times = 0
        while length_all_demos > max_length_all_demos:  # repeat sampling untill the suitable prefix is sampled
            print_rank(f"{self.split} | {data_name} | {prompt_name} | all demos lengths too large {length_all_demos} > {max_length_all_demos}")
            print(balance)
            if balance and DATA_CONFIG[data_name].finit_label(prompt_name):
                demo_idxs = []
                for l in pool_l:
                    demo_idxs.extend(self.rng_sample.sample(pool_l[l], k=int(shot / len(pool_l))))
            else:
                demo_idxs = self.rng_sample.sample(pool_sid, k=shot)
            repeat_times += 1
            if repeat_times > 10:
                shot = shot - 1

            length_all_demos = sum([len(pool_d[idx]["context_ids"]) + len(pool_d[idx]["target_ids"]) for idx in demo_idxs])
        
        # print("Shots", origin_shot, shot)
        # if origin_shot != 0 and len(demo_idxs) == 0:
        #     # add samples for shot == 0
        #     if balance and DATA_CONFIG[data_name].finit_label(prompt_name):
        #         demo_idxs = []
        #         for l in pool_l:
        #             demo_idxs.append(self.rng_sample.choice(pool_l[l]))
        #             self.rng_sample.shuffle(demo_idxs)
        #             length_all_demos = sum([len(pool_d[idx]["context_ids"]) + len(pool_d[idx]["target_ids"]) for idx in demo_idxs])
        #             while length_all_demos > max_length_all_demos:
        #                 print(len(demo_idxs), length_all_demos, max_length_all_demos)
        #                 demo_idxs.pop()
        #                 length_all_demos = sum([len(pool_d[idx]["context_ids"]) + len(pool_d[idx]["target_ids"]) for idx in demo_idxs])

        self.rng_order.shuffle(demo_idxs)

        return demo_idxs

    def build_icl_demos_share(self, demo_pool, label_idx_pool, balance=False):
        assert self.split != "train"
        log_str = f"{self.split} Sample icl train data share"
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        
        self.demo_pool = demo_pool # for __getitem__

        for data_name in self.data:
            for prompt_name in self.data[data_name]:
                
                if self.args.reset_seed_each_data:
                    self.rng_sample.seed(self.args.seed)
                    self.rng_order.seed(self.args.seed_order)
                
                d = self.data[data_name][prompt_name]
                pool_d = demo_pool[data_name][prompt_name]
                pool_l = label_idx_pool[data_name][prompt_name]
                pool_sid = [sample["idxs"] for sample in pool_d]
                
                if len(d) == 0:
                    continue

                max_length_all_demos = self._get_max_demo_length_share(d)

                demo_idxs = self._get_demo_ids_share(data_name, prompt_name, pool_d, pool_l, pool_sid, max_length_all_demos, balance)
                all_demo_ids = [pool_d[idx]["context_ids"] + pool_d[idx]["target_ids"] for idx in demo_idxs]
                
                if self.args.add_bos:
                    flat_all_demo_ids = [self.tokenizer.bos_token_id] + [x for y in all_demo_ids for x in y[1:]]
                else:
                    flat_all_demo_ids = [x for y in all_demo_ids for x in y]
                    
                print_rank(self.tokenizer.decode(flat_all_demo_ids).encode("utf-8"))
                
                valid_max_length = 0
                valid_max_length_per_sample = 0
                valid_max_all_context_length = 0
                for samp in d:
                    samp_length = len(samp["context_ids"]) + (max(map(len, samp["options_ids"])) if samp["options_ids"] is not None else len(samp["target_ids"]))
                    tot_length = len(flat_all_demo_ids) + samp_length
                    all_context_length = len(flat_all_demo_ids) + len(samp["context_ids"])

                    valid_max_length = max(tot_length, valid_max_length)
                    valid_max_length_per_sample = max(samp_length, valid_max_length_per_sample)
                    valid_max_all_context_length = max(all_context_length, valid_max_all_context_length)
                    samp.update({
                        "demo_idxs": demo_idxs
                    })

                self.valid_max_length[data_name][prompt_name] = valid_max_length
                self.valid_max_length_per_sample[data_name][prompt_name] = valid_max_length_per_sample
                self.valid_max_length_all_demos[data_name][prompt_name] = len(flat_all_demo_ids)
                self.valid_max_length_all_context[data_name][prompt_name] = valid_max_all_context_length
                
                label_dist = Counter([pool_d[idx]["target_str"] for idx in demo_idxs]) if balance and DATA_CONFIG[data_name].finit_label(prompt_name) else None
                log_str = f"{self.split} | {data_name} | {prompt_name} | valid max length: {valid_max_length} | label dist: {label_dist}"
                print_rank(log_str)
                print_rank("#" * 100)
                save_rank(log_str, os.path.join(self.args.save, "log.txt"))


class ICLTrainDataset(ICLDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=True, build_train_idx=True):
        super(ICLTrainDataset, self).__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool)
        if build_train_idx:
            self.iteration_idxs = self.build_iteration_index()

    def build_iteration_index(self):
        epochs = self.args.epochs
        idxs = [[] for _ in range(epochs)]
        log_str = f"{self.split} | build training index"
        print_rank(log_str)
        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
        for data_name in self.data:
            for prompt_name in self.data[data_name]:
                d = self.data[data_name][prompt_name]
                if len(d) == 0:
                    continue
                sample_num = min(self.args.flan_sample_max, len(d))
                data_idx = [i for i in range(len(d))]
                repeat_num = math.ceil(epochs * sample_num / len(data_idx))
                total_data_idx = []
                for i in range(repeat_num):
                    np.random.shuffle(data_idx)
                    total_data_idx.extend(data_idx) # 第一遍没采到的第二遍可以采到
                
                for e in range(epochs):
                    l = total_data_idx[e*sample_num:(e+1)*sample_num]
                    idxs[e].extend([(data_name, prompt_name, x) for x in l])
                
                log_str = f"{data_name} | {prompt_name} | repeat num: {repeat_num} | sample num: {sample_num} | data_idx len: {len(data_idx)} | total_data_idx: {len(total_data_idx)}"
                print_rank(log_str)
                save_rank(log_str, os.path.join(self.args.save, "log.txt"))

        first_len = len(idxs[0])
        for e, x in enumerate(idxs):
            assert len(x) == first_len, (e, len(x), first_len)

        return idxs

    def __len__(self):
        return len(self.iteration_idxs[0])

    def __getitem__(self, idx):
        data_name, prompt_name, sid = self.iteration_idxs[self.cur_epoch][idx]
        d = self.data[data_name][prompt_name][sid]
        ids = self._from_demo_idx_to_ids(data_name, prompt_name, d)
        new_d = {
            **d,
            **ids
        }
        return new_d
    
    def show_example(self):
        pass
    
    def move_to_device(self, model_data, no_model_data, device):
        for k in model_data:
            if model_data[k] is not None:
                model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            if isinstance(no_model_data[k], torch.Tensor):
                no_model_data[k] = no_model_data[k].to(device)
    
    
class ICLEvalDataset(ICLDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=False):
        super(ICLEvalDataset, self).__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool=as_pool)
        self.data_name = None
        self.prompt_name = None
    
    def get_metrics(self):
        return self.data_prompts[self.data_name][self.prompt_name].metadata.metrics
    
    def set_name_prompt(self, data_name, prompt_name):
        self.data_name = data_name
        self.prompt_name = prompt_name

    def label_size(self):
        return len(self.label_sid_map[self.data_name][self.prompt_name])

    def current_data(self):
        return self.data[self.data_name][self.prompt_name]

    def __len__(self):
        return len(self.data[self.data_name][self.prompt_name])

    def __getitem__(self, idx):
        d = self.data[self.data_name][self.prompt_name][idx]
        ids = self._from_demo_idx_to_ids(self.data_name, self.prompt_name, d)
        new_d = {
            **d,
            **ids
        }
        return new_d
    
    def move_to_device(self, model_data, no_model_data, device):
        for k in model_data:
            if model_data[k] is not None:
                model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            if isinstance(no_model_data[k], torch.Tensor):
                no_model_data[k] = no_model_data[k].to(device)
