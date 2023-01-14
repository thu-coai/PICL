import random
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import re
import json
import h5py
from tqdm import tqdm
from .distributed_indexed import DistributedMMapIndexedDataset
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.distributed import get_rank, get_world_size
from icl_train.utils import print_rank_hf as print_rank


def _collate_demo_in_model(args, all_demo_ids_full_batch, bs, max_length_per_sample, max_length_all_demos, chunk_len, pad_id):
    num_all_demos = sum(map(len, all_demo_ids_full_batch))
    demo_batch = {
        "select_index": torch.zeros(bs, max_length_all_demos, dtype=torch.long),
        "context_select_index": torch.zeros(bs, max_length_all_demos, dtype=torch.long),
        "select_index_inv": [],
        "max_length_all_demos": max_length_all_demos,
        "demo_pos_shift": [],
        "test_pos_shift": [],
        "compress_input_ids": torch.ones(bs, max_length_all_demos, dtype=torch.long) * pad_id,
        "compress_position_ids": torch.zeros(bs, max_length_all_demos, dtype=torch.long),
        "attention_mask": torch.zeros(num_all_demos, chunk_len, chunk_len),
    }
    n_demo = 0
    for i, all_demo_ids in enumerate(all_demo_ids_full_batch):
        st = 0
        max_demo_pos = max(map(len, all_demo_ids), default=0)
        # print(max_demo_pos)

        # in model many
        _select_index = []
        _context_select_index = []
        for ids in all_demo_ids:
            demo_batch["compress_input_ids"][i][st:st+len(ids)] = torch.tensor(ids, dtype=torch.long)
            if args.pos_type in [0,1]:
                demo_batch["compress_position_ids"][i][st:st+len(ids)] = torch.arange(0, len(ids), dtype=torch.long)
            elif args.pos_type == 2:
                demo_batch["compress_position_ids"][i][st:st+len(ids)] = torch.arange(max_demo_pos-len(ids), max_demo_pos, dtype=torch.long)
            elif args.pos_type == 3:
                demo_batch["compress_position_ids"][i][st:st+len(ids)] = torch.arange(max_demo_pos-len(ids), max_demo_pos, dtype=torch.long)
                demo_batch["compress_position_ids"][i][st] = 0
            else:
                raise NotImplementedError
            
            demo_batch["attention_mask"][n_demo][:len(ids),:len(ids)] = torch.tril(torch.ones(len(ids), len(ids)))
            
            _select_index.extend([n_demo*chunk_len + idx for idx in range(len(ids))])
            _context_select_index.extend([n_demo*chunk_len + idx for idx in range(0,len(ids))])
            demo_batch["select_index_inv"].append([i*max_length_all_demos+st+idx for idx in range(len(ids))] + [0] * (chunk_len - len(ids)))
            
            # other
            st += len(ids)
            demo_batch["demo_pos_shift"].append(max_demo_pos)
            n_demo += 1
        
        demo_batch["test_pos_shift"].append(max_demo_pos)
        demo_batch["select_index"][i][:len(_select_index)] = torch.tensor(_select_index, dtype=torch.long)
        demo_batch["context_select_index"][i][:len(_context_select_index)] = torch.tensor(_context_select_index, dtype=torch.long)
        
        assert max_demo_pos + max_length_per_sample < args.gpt_max_length, (max_demo_pos, max_length_per_sample) # max position ids

    demo_batch["select_index_inv"] = torch.tensor(demo_batch["select_index_inv"], dtype=torch.long)

    inner_inputs = {
        "demo_input_ids": demo_batch["compress_input_ids"],
        "demo_index": demo_batch["select_index"],
        "demo_context_index": demo_batch["context_select_index"],
        "demo_index_inv": demo_batch["select_index_inv"],
        "demo_attention_mask": demo_batch["attention_mask"],
        "demo_position_ids": demo_batch["compress_position_ids"],
    }

    return demo_batch, inner_inputs


class ICLUnsupTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path_icl, path_lm, split, num, lm_num, ratio, shot, mode, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.shot = shot
        self.pad_id = self.tokenizer.eos_token_id
        self.num = num
        self.lm_num = lm_num
        self.ratio = ratio
        self.max_length = args.max_length
        self.max_length_per_sample = args.max_length_per_sample
        self.max_length_all_demos = args.max_length_all_demos
        self.rng_sample = rng_sample
        self.mode = mode
        
        self.delimiter_id = {
            "<n>": 198,
            "<eos>": 50256,
            "2<n>": 628
        }[self.args.end_token]
        if mode in ["icl", "mixed"]:
            self.icl_ctx = DistributedMMapIndexedDataset(path_icl, f"{split}_{args.unsup_data_name}", get_rank(), get_world_size())
            if self.num > 0:
                self.icl_data_len = min(len(self.icl_ctx), self.num)
            else:
                self.icl_data_len = len(self.icl_ctx)
            print_rank(f"Total ICL instances: {len(self.icl_ctx)}. Train ICL instances: {self.icl_data_len}")
        
        if mode in ["lm", "mixed"]:
            self.lm_ctx = DistributedMMapIndexedDataset(path_lm, f"{split}_{args.lm_data_name}", get_rank(), get_world_size())
            self.lm_data_len = len(self.lm_ctx)
            if mode == "lm":
                if self.lm_num > 0:
                    self.lm_data_len = min(len(self.lm_ctx), self.lm_num)
                else:
                    self.lm_data_len = len(self.lm_ctx)
            else:
                if self.args.lm_ratio is None or int(self.args.lm_ratio * self.icl_data_len) >= len(self.lm_ctx):
                    self.lm_data_len = len(self.lm_ctx)
                else:
                    self.lm_data_len = int(self.args.lm_ratio * self.icl_data_len)
            print_rank(f"Total LM instances: {len(self.lm_ctx)}. Train LM instances: {self.lm_data_len}")
        
        if mode == "mixed":
            print_rank(f"ICL/LM: {self.icl_data_len}/{self.lm_data_len}")                

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "icl":
            return self.icl_data_len
        elif self.mode == "lm":
            return self.lm_data_len
        else:
            return self.icl_data_len + self.lm_data_len
   
    def __getitem__(self, index):
        if self.mode == "icl":
            return self._get_icl(index)
        if self.mode == "lm":
            return self._get_lm(index)
        elif self.mode == "mixed":
            if index >= self.icl_data_len:
                return self._get_lm(index-self.icl_data_len)
            else:
                return self._get_icl(index)
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }
    
    def _get_icl(self, index):
        data = self.icl_ctx[index]
        input_ids = data.astype(int)
        split_input_ids = self._split_input_ids(input_ids)
        demos = split_input_ids[:-1]
        demos = demos[:self.shot]
        random.shuffle(demos)
        
        test_input_ids = split_input_ids[-1]
        assert test_input_ids[-1] == self.delimiter_id
        
        return {
            "demo_ids": demos,
            "test_input_ids": split_input_ids[-1]
        }
        
    def show_example(self):
        if dist.get_rank() == 0:
            with open(os.path.join(self.args.save, "example.txt"), "w") as f:
                if self.mode in ["mixed", "icl"]:
                    f.write("ICL examples:\n")
                    for i in range(4):
                        f.write("##### Demo Samples #####\n")
                        tmp = self._get_icl(i)
                        f.write(str(tmp) + "\n")
                        for ids in tmp["demo_ids"]:
                            f.write(self.tokenizer.decode(ids) + "\n\n")
                        f.write("##### Test Sample #####\n")
                        f.write(self.tokenizer.decode(tmp["test_input_ids"]) + "\n")
                        f.write("#" * 50 + "\n")
                if self.mode in ["mixed", "lm"]:
                    f.write("\n" * 5)
                    f.write("LM examples:\n")
                    for i in range(4):
                        tmp = self._get_lm(i)
                        f.write(str(tmp) + "\n")
                        f.write(self.tokenizer.decode(tmp["input_ids"]) + "\n")
                        f.write("#" * 50 + "\n")
    
    def _split_input_ids(self, input_ids):
        assert input_ids[-1] == self.delimiter_id
        split_input_ids = []
        b, e = 0, 0
        while e < len(input_ids):
            if input_ids[e] == self.delimiter_id:
                split_input_ids.append(input_ids[b:e+1])
                b = e + 1
            e += 1
        return split_input_ids
    
    def _get_lcs_start(self, demo_ids, test_input_ids):
        l = 0
        while l < len(test_input_ids):
            if any([(np.array_equal(_dids[:l+1], test_input_ids[:l+1])) for _dids in demo_ids]):
                l += 1
            else:
                break
        if l >= len(test_input_ids):
            s = l - 1
        elif l + 1 >= len(test_input_ids):
            s = l
        else:
            s = l + 1  # from the second different token
        return s

    def _process_icl(self, i, samp, model_data, no_model_data):
        input_ids = samp["demo_ids"] + [samp["test_input_ids"]]
        input_ids = [x for y in input_ids for x in y]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1, :input_len-1] = torch.tril(torch.ones(input_len-1, input_len-1))
        model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        if self.args.icl_sup == "all_target":
            no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
            no_model_data["loss_mask"][i][:input_len-1] = 1.0
        elif self.args.icl_sup == "test_target":
            assert input_ids[-1] == self.delimiter_id
            shift = 0 if len(samp["demo_ids"]) == 0 else 1
            no_model_data["label"][i][input_len-len(samp["test_input_ids"])-shift:input_len-1] = torch.tensor(samp["test_input_ids"][1-shift:], dtype=torch.long)
            no_model_data["loss_mask"][i][input_len-len(samp["test_input_ids"])-shift:input_len-1] = 1.0
        elif self.args.icl_sup == "test_target_sub":
            # s the start supervision place
            s = self._get_lcs_start(samp["demo_ids"], samp["test_input_ids"])
            if self.args.sup_start_pos is not None:
                s = max(s, self.args.sup_start_pos)
            else:
                assert self.args.sup_start_ratio is not None
                s = max(s, int(self.args.sup_start_ratio * len(samp["test_input_ids"])))
            demo_lens = sum(map(len, samp["demo_ids"]))
            no_model_data["label"][i][demo_lens+s-1:input_len-1] = torch.tensor(samp["test_input_ids"][s:], dtype=torch.long)
            no_model_data["loss_mask"][i][demo_lens+s-1:input_len-1] = 1.0
        elif self.args.icl_sup == "test_target_lcs":
            # s the start supervision place
            s = self._get_lcs_start(samp["demo_ids"], samp["test_input_ids"])
            demo_lens = sum(map(len, samp["demo_ids"]))
            no_model_data["label"][i][demo_lens+s-1:input_len-1] = torch.tensor(samp["test_input_ids"][s:], dtype=torch.long)
            no_model_data["loss_mask"][i][demo_lens+s-1:input_len-1] = 1.0
        else:
            raise NotImplementedError

    def _process_lm(self, i, samp, model_data, no_model_data):
        input_ids = samp["input_ids"]
        input_len = len(input_ids)
        model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        model_data["attention_mask"][i][:input_len-1, :input_len-1] = torch.tril(torch.ones(input_len-1, input_len-1))
        model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        no_model_data["loss_mask"][i][:input_len-1] = 1.0

    def collate(self, samples):
        bs = len(samples)

        max_length = self.max_length
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, max_length),
            "position_ids": torch.zeros(bs, max_length, dtype=torch.long),
        }
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }

        for i, samp in enumerate(samples):
            if "demo_ids" in samp:
                self._process_icl(i, samp, model_data, no_model_data)
            else:
                self._process_lm(i, samp, model_data, no_model_data)
        
        return model_data, no_model_data
    
    def collate_zs(self, samples):
        bs = len(samples)
                
        model_data = {
            "input_ids": torch.ones(bs, self.max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length, self.max_length),
            "position_ids": torch.zeros(bs, self.max_length, dtype=torch.long),
        }
        no_model_data = {
            "label": torch.ones(bs, self.max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, self.max_length)
        }
        
        for i, samp in enumerate(samples):
            tmp_input_ids = samp["demo_ids"] + [samp["test_input_ids"]]
            input_ids = [x for y in tmp_input_ids for x in y]
            input_len = len(input_ids)
            model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
            s = 0
            for ids in tmp_input_ids:
                model_data["attention_mask"][i][s:s+len(ids)-1, s:s+len(ids)-1] = torch.tril(torch.ones(len(ids)-1, len(ids)-1))
                model_data["position_ids"][i][s:s+len(ids)-1] = torch.arange(0, len(ids)-1, dtype=torch.long)
                no_model_data["label"][i][s:s+len(ids)-1] = torch.tensor(ids[1:], dtype=torch.long)
                no_model_data["loss_mask"][i][s:s+len(ids)-1] = 1.0
                s += len(ids)

        return model_data, no_model_data

    def collate_unordered(self, samples):
        bs = len(samples)

        model_data = {
            "input_ids": torch.ones(bs, self.max_length_all_demos, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length_all_demos, self.max_length_all_demos),
            "position_ids": torch.zeros(bs, self.max_length_all_demos, dtype=torch.long),
        }
        no_model_data = {
            "label": torch.ones(bs, self.max_length_all_demos, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, self.max_length_all_demos)
        }

        for i, samp in enumerate(samples):
            input_len = len(samp["input_ids"]) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(samp["input_ids"][:-1], dtype=torch.long)
            split_input_ids = self._split_input_ids(samp["input_ids"])
            icl_train_data_ids = split_input_ids[:-1]
            target_ids = split_input_ids[-1]
            st = 0
            max_demo_len = max(map(len, icl_train_data_ids), default=0)
            for demo_ids in icl_train_data_ids:
                model_data["attention_mask"][i][st:st+len(demo_ids),st:st+len(demo_ids)] = torch.tril(torch.ones(len(demo_ids), len(demo_ids)))
                if self.args.pos_type == 0:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(0, len(demo_ids), dtype=torch.long)
                elif self.args.pos_type == 1:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                elif self.args.pos_type == 2:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                elif self.args.pos_type == 3:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                    model_data["position_ids"][i][st] = 0
                st += len(demo_ids)
            model_data["attention_mask"][i][st:input_len, 0:st] = 1
            model_data["attention_mask"][i][st:input_len, st:input_len] = torch.tril(torch.ones(input_len - st, input_len - st))
            if self.args.pos_type == 0:
                model_data["position_ids"][i][st:input_len] = torch.arange(0, input_len-st, dtype=torch.long)
            elif self.args.pos_type in [1,2,3]:
                model_data["position_ids"][i][st:input_len] = torch.arange(0, input_len-st, dtype=torch.long) + max_demo_len

            if self.args.icl_sup == "test_target":
                no_model_data["label"][i][input_len-len(target_ids)+1:input_len] = torch.tensor(target_ids[1:], dtype=torch.long)
                no_model_data["loss_mask"][i][input_len-len(target_ids)+1:input_len] = 1.0
            else:
                raise NotImplementedError
        
        return model_data, no_model_data

    def collate_many(self, samples):
        bs = len(samples)
        model_batch = {
            "input_ids": torch.ones(bs, self.max_length_per_sample, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length_per_sample, self.max_length_per_sample + self.max_length_all_demos, dtype=torch.long),
            "position_ids": torch.zeros(bs, self.max_length_per_sample, dtype=torch.long),
        }
        no_model_batch = {
            "label": torch.ones(bs, self.max_length_per_sample, dtype=torch.long) * (-100),
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "loss_mask": torch.zeros(bs, self.max_length_per_sample)
        }
        chunk_len = self.max_length_per_sample
        
        demo_batch, inner_inputs = _collate_demo_in_model(self.args, [samp["demo_ids"] for samp in samples], bs, self.max_length_per_sample, self.max_length_all_demos, chunk_len, self.pad_id)
        
        assert len(demo_batch["test_pos_shift"]) == len(samples)
        for i, samp in enumerate(samples):
            input_ids = samp["test_input_ids"]
            input_len = len(samp["test_input_ids"]) - 1
            demos_len = sum([len(x) for x in samp["demo_ids"]])
            model_batch["input_ids"][i][:input_len] = torch.tensor(input_ids[:-1], dtype=torch.long)

            model_batch["attention_mask"][i][:input_len, :demos_len] = 1.0
            attn_mask = torch.tril(torch.ones(input_len, input_len))
            model_batch["attention_mask"][i][:input_len, self.max_length_all_demos:self.max_length_all_demos+input_len] = attn_mask

            if self.args.pos_type == 0:
                model_batch["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            elif self.args.pos_type in [1,2]:
                model_batch["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
                model_batch["position_ids"][i][:input_len] += demo_batch["test_pos_shift"][i]
            elif self.args.pos_type == 3:
                model_batch["position_ids"][i][1:input_len] = torch.arange(0, input_len-1, dtype=torch.long)
                model_batch["position_ids"][i][1:input_len] += demo_batch["test_pos_shift"][i]
                model_batch["position_ids"][i][0] = 0
            else:
                raise NotImplementedError()
            if self.args.icl_sup == "test_target":
                no_model_batch["label"][i][:input_len] = torch.tensor(input_ids[1:], dtype=torch.long)
            else:
                raise NotImplementedError
            no_model_batch["loss_mask"][i][:input_len] = 1.0
        
        return model_batch, no_model_batch, demo_batch, inner_inputs
    
    def move_to_device(self, model_data, no_model_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)
        
        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

    def set_h5(self, path, name):
        with h5py.File(path, "w") as f:
            f.create_dataset(name, data=np.zeros(
                (0, self.max_length), dtype=np.float16), maxshape=(None, None), chunks=True)

    def dump_h5(self, path, name, embeddings):
        assert os.path.exists(path)
        with h5py.File(path, "a") as f:
            d = f[name]
            d.resize(d.shape[0] + embeddings.shape[0], axis=0)
            d[-len(embeddings):] = embeddings
        # return d.shape[0] + embeddings.shape[0]

    def sum_h5(self, path, name):
        with h5py.File(path, "r+") as f:
            s_origin = f[name].shape
            f[name].resize(self.__len__(), axis=0)
            s_resize = f[name].shape
            print(f"Dumped to {path}, origin size = {s_origin}, resize = {s_resize}")


class SmallDataset(Dataset):
    def __init__(self, args, tokenizer, path, rng_sample):
        self.args = args
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.pad_id = tokenizer.eos_token_id
        self.shot = args.shot
        self.rng_sample = rng_sample
        
        self.data = self.process(path)

    def clean(self, s):
        s = re.sub(r"\s+", " ", s)
        return s

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _pack(self, demo_ids, test_input_ids):
        demo_ids = demo_ids[:self.shot]
        new_demo_ids = []
        for demo_id in demo_ids:
            if sum(map(len, new_demo_ids)) + len(demo_id) + len(test_input_ids) < self.max_length:
                new_demo_ids.append(demo_id)
            else:
                break
        self.rng_sample.shuffle(new_demo_ids)
        return new_demo_ids

    def process(self, path):
        with open(path, "r") as f:
            lines = json.load(f)
        data = []
        for line in lines:
            demo_ids = [self.tokenizer.encode(self.clean(demo_id) + '\n') for demo_id in line[1]]
            test_input_ids = self.tokenizer.encode(self.clean(line[0]) + "\n")
            demo_ids = self._pack(demo_ids, test_input_ids)
            data.append({
                "demo_ids": demo_ids,
                "test_input_ids": test_input_ids
            })
        return data

    def move_to_device(self, model_data, no_model_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

    def collate(self, samples):
        bs = len(samples)
        
        max_length = self.max_length
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, max_length),
            "position_ids": torch.zeros(bs, max_length, dtype=torch.long),
        }
        no_model_data = {
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }

        for i, samp in enumerate(samples):
            input_ids = samp["demo_ids"] + [samp["test_input_ids"]]
            input_ids = [x for y in input_ids for x in y]
            input_len = len(input_ids)
            model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
            model_data["attention_mask"][i][:input_len-1, :input_len-1] = torch.tril(torch.ones(input_len-1, input_len-1))
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
            if self.args.icl_sup == "all_target":
                no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
                no_model_data["loss_mask"][i][:input_len-1] = 1.0
                
        return model_data, no_model_data


if __name__ == "__main__":
    from model_center.arguments import get_args
    from transformers import GPT2Tokenizer
    from torch.utils.data import DataLoader, DistributedSampler
    import deepspeed
    from icl_train.utils import set_random_seed

    def init_distributed(args):
        args.rank = int(os.getenv("RANK", "0"))
        args.world_size = int(os.getenv("WORLD_SIZE", "1"))
        args.local_rank = int(os.getenv("LOCAL_RANK", "0"))

        if args.rank == 0:
            print(f"using world size: {args.world_size}")

        # Manually set the device ids.
        device = args.rank % torch.cuda.device_count()
        if args.local_rank is not None:
            device = args.local_rank
        torch.cuda.set_device(device)

        deepspeed.init_distributed()


    def initialize():
        # get arguments
        args = get_args()
        # init bmt
        init_distributed(args)
        set_random_seed(args.seed)
        # init save folder
        if args.save != None:
            os.makedirs(args.save, exist_ok=True)
        return args

    args = initialize()
    args.max_length = 1024
    args.max_length_per_sample = 256
    args.data_dir = "/home/lidong1/CodeRepo/icl_train/unsup_data/general/10M_256/rand_icl/256_1024_-1/r2s<n>/"
    args.unsup_data_name = "tokenized"
    tokenizer = GPT2Tokenizer.from_pretrained("/home/lidong1/CodeRepo/icl_train/results/gpt2-large")
    dataset = ICLUnsupTrainDataset(args, tokenizer, args.data_dir, None, "train", -1, 1, 16, None)
    
    # sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=7, num_workers=1, collate_fn=dataset.collate)
    
    # for e in range(5):
    #     print_rank("=" * 10, e, "=" * 10)
    #     sampler.set_epoch(e)
    #     for batch in dataloader:
    #         if dist.get_rank() == 0:
    #             print(tokenizer.decode(batch["input_ids"][0].tolist()))
    #         exit(0)

    num_demos = []

    for i, d in enumerate(tqdm(dataset)):
        num_demos.append(len(d["demo_ids"]))
        
        if i > 10000:
            break
        
    print(np.mean(num_demos))
    
    plt.hist(num_demos, bins=20)
    plt.savefig(os.path.join(args.data_dir, "demo_dist.png"))