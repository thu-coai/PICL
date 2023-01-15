
import os
import random

import numpy as np
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import get_rank, barrier


def print_args(args):
    """Print arguments."""

    print('arguments:', flush=True)
    for arg in vars(args):
        dots = '.' * (29 - len(arg))
        print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)


def save_rank(log_str, save_path, rank=0):
    if not dist.is_initialized() or dist.get_rank() == rank:
        with open(save_path, "a") as f:
            f.write(log_str + "\n")


def print_rank(*args, rank=0, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == rank:
        print(*args, **kwargs)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'ln_f.weight', 'ln_1.weight', 'ln_2.weight', 'ln_cross_attn']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)]},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    return optimizer_grouped_parameters


class MultiPromptResults():
    def __init__(self, data_prompts):
        self.res = defaultdict(lambda:defaultdict(lambda:{}))
        self.preds = defaultdict(lambda: defaultdict(lambda: ()))
        self.losses = defaultdict(lambda: defaultdict(lambda: 0))
        self.tot_losses = defaultdict(lambda: defaultdict(lambda: 0))
        
        self.data_prompts = data_prompts

    def add_res(self, data_name, prompt_name, item):
        self.res[data_name][prompt_name].update(item)
        
    def add_preds(self, data_name, prompt_name, item):
        self.preds[data_name][prompt_name] = item
    
    def add_loss(self, data_name, prompt_name, item):
        self.losses[data_name][prompt_name] = item

    def add_tot_loss(self, data_name, prompt_name, item):
        self.tot_losses[data_name][prompt_name] = item

    def average(self, key="res"):
        return np.mean([self.average_per_data(dn, key) for dn in self.res])
    
    def average_per_data(self, dn, key="res") -> Dict:
        return np.mean([self.average_per_prompt(dn, pn, key) for pn in self.res[dn]])
    
    def average_per_prompt(self, dn, pn, key="res"):
        if key == "res":
            return np.mean(list(self.res[dn][pn].values()))
        elif key == "loss":
            return self.losses[dn][pn]
        elif key == "tot_loss":
            return self.tot_losses[dn][pn]
        else:
            raise ValueError(f"{key} not support")
    
    def get_res(self, data_name, prompt_name, key="res"):
        if key == "res":
            return self.res[data_name][prompt_name]
        elif key == "loss":
            return self.losses[data_name][prompt_name]
        elif key == "tot_loss":
            return self.tot_losses[data_name][prompt_name]
        else:
            raise ValueError(f"{key} not support")
            
    def all_res(self, data_name=None, prompt_name=None, key="res"):
        if data_name is None:
            return {dn: {pn: self.get_res(dn, pn, key) for pn in self.res[dn]} for dn in self.res}
        elif prompt_name is None:
            return {pn: self.get_res(data_name, pn, key) for pn in self.res[data_name]}
        else:
            return self.get_res(data_name, prompt_name, key)
    
    def all_data_names(self):
        return list(self.res.keys())
    
    def save_res(self, save_dir, step):
        if get_rank() == 0:
            save_dir = os.path.join(save_dir, "preds", str(step))
            os.makedirs(save_dir, exist_ok=True)
            for dn in self.res:
                with open(os.path.join(save_dir, f"{dn}.txt"), "w") as f:
                    f.write(str(self.average_per_data(dn)) + " | " + str(self.average_per_data(dn, key="loss")) + " | " + str(self.average_per_data(dn, key="tot_loss")) + "\n\n")
                    for pn in self.res[dn]:
                        f.write(pn + " | " + str(self.res[dn][pn]) + " | " + str(self.losses[dn][pn]) + " | " + str(self.tot_losses[dn][pn]) + " | " + self.data_prompts[dn][pn].jinja.replace("\n", "\t\t") + "\n")
                        for l, p in zip(self.preds[dn][pn][0], self.preds[dn][pn][1]):
                            f.write((str(l.strip()) + "\t\t" + str(p.strip()) + "\n").encode("utf-8").decode("latin1"))
                        f.write("\n")
                    f.write("\n\n")
                    
                    
