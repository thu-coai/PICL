import random
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import h5py
from .distributed_indexed import DistributedMMapIndexedDataset

import torch.distributed as dist
from torch.distributed import get_rank, get_world_size
from utils import print_rank


class ICLPretrainDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        path_icl,
        split="train",
        num=-1,
        lm_num=0,
        shot=16,
        mode="icl",
        path_lm=None,
        path_icl_idx=None,
        rng_sample: random.Random = None):
        
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.shot = shot
        self.pad_id = self.tokenizer.eos_token_id
        self.num = num
        self.lm_num = lm_num
        self.max_length = args.max_length
        self.max_length_per_sample = args.max_length_per_sample
        self.rng_sample = rng_sample
        self.delimiter_id = 198
        self.mode = mode
        
        if path_icl_idx is None:
            path_icl_idx = path_icl

        if mode in ["icl", "merge"]:
            self.icl_ctx = DistributedMMapIndexedDataset(path_icl, args.picl_data_name, get_rank(), get_world_size()) # for new idx
            self.icl_idx = DistributedMMapIndexedDataset(path_icl_idx, f"{split}_icl", get_rank(), get_world_size()) # for origin idx
            with h5py.File(os.path.join(path_icl, "map.h5")) as h5f:
                self.icl_idx_map = h5f["map_o2n"][:]

            if self.num > 0:
                self.icl_data_len = min(len(self.icl_idx), num)
            else:
                self.icl_data_len = len(self.icl_idx)
            print_rank(f"Total ICL instances: {len(self.icl_idx)}. Train ICL instances: {self.icl_data_len}")

        if mode in ["lm", "merge"]:
            self.lm_ctx = DistributedMMapIndexedDataset(path_lm, f"{split}_{args.lm_data_name}", get_rank(), get_world_size())
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
        
        if mode == "merge":
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
        elif self.mode == "lm":
            return self._get_lm(index)
        elif self.mode == "merge":
            if index >= self.icl_data_len:
                return self._get_lm(index-self.icl_data_len)
            else:
                return self._get_icl(index)
        else:
            raise NotImplementError()
    
    def _get_lm(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }
    
    def _get_icl(self, index):
        icl_indices = self.icl_idx[index].astype(int)
        # idx after unordered preprocess may be in random order icl_idx_map: from origin idx to new idx
        data = [self.icl_ctx[int(self.icl_idx_map[i])].tolist() for i in icl_indices]   
                
        q_id = data[0][:self.max_length_per_sample-1] + [self.delimiter_id]
        r_ids = [rr[:self.max_length_per_sample-1] + [self.delimiter_id] for rr in data[1:]]

        while len(q_id) + sum(map(len, r_ids)) >= self.max_length and len(r_ids) > 0:
            r_ids.pop()
            
        r_ids = r_ids[:self.shot]
        self.rng_sample.shuffle(r_ids)
        
        return {
            "demo_ids": r_ids,
            "test_input_ids": q_id
        }
 
    def show_example(self):
        if dist.get_rank() == 0:
            if self.mode in ["merge", "icl"]:
                with open(os.path.join(self.args.save, "example_icl.txt"), "w") as f:   
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
            if self.mode in ["merge", "lm"]:
                with open(os.path.join(self.args.save, "example_lm.txt"), "w") as f:   
                    f.write("\n" * 5)
                    f.write("LM examples:\n")
                    for i in range(4):
                        tmp = self._get_lm(i)
                        f.write(str(tmp) + "\n")
                        f.write(self.tokenizer.decode(tmp["input_ids"]) + "\n")
                        f.write("#" * 50 + "\n")

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
