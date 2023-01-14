import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from data_utils.distributed_indexed import DistributedMMapIndexedDataset

import torch.distributed as dist
from torch.distributed import get_rank, get_world_size
from utils import print_rank_hf as print_rank
from utils import save_rank_hf as save_rank
from utils import set_random_seed, print_args
import deepspeed
import time
from model_center.arguments import get_args

from transformers import GPT2LMHeadModel, GPT2Tokenizer


class ICLUnsupTrainDataset(Dataset):
    def __init__(self, args, tokenizer, path_lm, split, num, ratio, shot, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.shot = shot
        self.pad_id = self.tokenizer.eos_token_id
        self.num = num
        self.ratio = ratio
        self.max_length = 1024
        self.rng_sample = rng_sample
        self.delimiter_id = {
            "<n>": 198,
            "<eos>": 50256,
            "2<n>": 628
        }[self.args.end_token]
        self.lm_ctx = DistributedMMapIndexedDataset(path_lm, f"valid_lm", get_rank(), get_world_size())                

    def __len__(self):
        return len(self.lm_ctx)
    
    def __getitem__(self, index):
        data = self.lm_ctx[index]
        input_ids = data.astype(int)
        return {
            "input_ids": input_ids
        }   

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
            self._process_lm(i, samp, model_data, no_model_data)
        
        return model_data, no_model_data
    
    def move_to_device(self, model_data, no_model_data, device):
        for k in model_data:
            model_data[k] = model_data[k].to(device)
        
        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)


def evaluate(args, tokenizer, model, dataset: ICLUnsupTrainDataset, split, epoch, mode, device):
    
    collate_fn = dataset.collate
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    model.eval()
    all_loss = 0.0
    step = 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            if it == 0:
                if dist.get_rank() == 0:
                    for ids in model_batch["input_ids"]:
                        print("#" * 20)
                        print(ids)
                        print(tokenizer.decode(ids))
                        print("#" * 20)

            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch).logits
            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            all_loss += loss.item()
            step += 1
    
    avg_loss = all_loss / step
    
    avg_loss = torch.tensor([avg_loss], device=device)
    gathered_loss = [torch.zeros_like(avg_loss, device=device) for _ in range(get_world_size())]
    dist.all_gather(gathered_loss, avg_loss)
    avg_loss = sum(gathered_loss) / len(gathered_loss)
    
    log_str = f"{mode} | {split} | avg_loss: {avg_loss}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
    
    return avg_loss


def prepare_dataset(args, tokenizer, rank, world_size):
    data = {}
    rng_sample = random.Random(args.seed)


    data["test"] = ICLUnsupTrainDataset(args, tokenizer, args.lm_data_dir, "valid", args.dev_num, args.dev_ratio, args.shot, rng_sample)

    return data


def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)
    
    return tokenizer


def get_model(args, device):
    model = GPT2LMHeadModel.from_pretrained(args.model_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def setup_model_and_optimizer(args, ds_config, device):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    optimizer, lr_scheduler = None, None
        
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model, optimizer, lr_scheduler


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


def main():
    torch.backends.cudnn.enabled = False
    
    args = initialize()
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    tokenizer = get_tokenizer(args)
    
    dataset = prepare_dataset(
        args,
        tokenizer,
        dist.get_rank(), dist.get_world_size(),
    )
    
    model, _, _ = setup_model_and_optimizer(args, ds_config, device)
        
    loss = evaluate(args, tokenizer, model, dataset["test"], "test", 0, "lm", device)


if __name__ == "__main__":
    print(os.environ["CODE_BASE"])
    main()
