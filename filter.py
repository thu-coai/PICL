import sys
if "/home/lidong1/.local/lib/python3.8/site-packages" in sys.path:
    sys.path.remove("/home/lidong1/.local/lib/python3.8/site-packages")

from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import os
import json
import time
import h5py
import numpy as np
import random
import deepspeed
from datetime import timedelta

from model_center import get_args

from data_utils.icl_unsup_train_datasets_fast import ICLUnsupTrainDataset

from utils import set_random_seed, print_args
from utils import print_rank_hf as print_rank
from utils import save_rank_hf as save_rank
from tqdm import tqdm

num_threads = 4
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
torch.set_num_threads(num_threads)


def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)

    return tokenizer


def get_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model_config)
    return model


def setup_model(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)

    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=None,
        args=args,
        lr_scheduler=None,
        config_params=ds_config
    )

    # get the memory usage
    # print_rank("Model mem\n", torch.cuda.memory_summary())
    return model


def prepare_dataset(args, tokenizer, rank, world_size):
    rng_sample = random.Random(args.seed)
    # if args.score_small:
    #     dataset = SmallDataset(args, tokenizer, args.data_dir, rng_sample)
    # else:
    dataset = ICLUnsupTrainDataset(args, tokenizer, args.data_dir, None, "valid", args.filter_num, 0, args.filter_ratio, args.shot, "icl", rng_sample)
    return dataset


def score_all(args, tokenizer, model, dataset: ICLUnsupTrainDataset, device):

    collate_fn = dataset.collate
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    model.eval()
    all_loss = 0.0
    step = 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch).logits
            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            all_loss += loss.item()
            step += 1
    
    avg_loss = all_loss / step
    
    return avg_loss


def score_small(args, tokenizer, model, dataset, device):

    collate_fn = dataset.collate

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    model.eval()
    all_avg_loss = []
    step = 0
    
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            
            # print(model_batch)
            # print(tokenizer.decode(model_batch["input_ids"][0].cpu().tolist()))
            # exit(0)
            
            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch).logits
            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            loss = loss.view(no_model_batch["loss_mask"].size())
            loss = loss * no_model_batch["loss_mask"]
            
            avg_loss = torch.sum(loss, dim=-1) / torch.sum(no_model_batch["loss_mask"], dim=-1)
            all_avg_loss.extend(avg_loss.cpu().tolist())
            
            gathered_losses = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss)
            gathered_losses = torch.stack(gathered_losses, dim=1).view(-1, loss.size(-1)).cpu().half().numpy()
            
            step += 1
    
    print(all_avg_loss)
    
    return np.mean(all_avg_loss)


def score(args, tokenizer, model, dataset: ICLUnsupTrainDataset, device, mode="icl"):

    print("Mode:", mode)
    
    collate_fn = dataset.collate if mode in ["icl", "lm"] else dataset.collate_zs

    # if mode == "lm":
    #     assert args.lm_ratio is not None and args.lm_data_dir is not None and args.lm_data_prefix is not None
    #     dataset.set_mode("lm")

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    model.eval()
    all_avg_loss = []
    step = 0

    ckpt_name = (args.ckpt_name).replace("/", "_")
    score_file_name = os.path.join(args.save, f"score_{mode}_{ckpt_name}.h5")

    if dist.get_rank() == 0:
        dataset.set_h5(score_file_name, "score")

    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):

            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch).logits
            loss = loss_func(
                logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            loss = loss.view(no_model_batch["loss_mask"].size())
            loss = loss * no_model_batch["loss_mask"]

            avg_loss = torch.sum(loss, dim=-1) / \
                torch.sum(no_model_batch["loss_mask"], dim=-1)
            all_avg_loss.extend(avg_loss.cpu().tolist())

            gathered_losses = [torch.zeros_like(
                loss) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_losses, loss)
            gathered_losses = torch.stack(
                gathered_losses, dim=1).view(-1, loss.size(-1)).cpu().half().numpy()

            # data_len = torch.zeros(1).to(device)
            if dist.get_rank() == 0:
                dataset.dump_h5(score_file_name, "score", gathered_losses)
            #     dist.scatter(data_len, [torch.tensor([l], dtype=torch.long).to(device) for _ in range(dist.get_world_size())], src=0)
            # else:
            #     dist.scatter(data_len, None, src=0)
            
            # if len(data_len) > args.filter_num:
            #     break

            step += 1

    if dist.get_rank() == 0:
        dataset.sum_h5(score_file_name, "score")

    return np.mean(all_avg_loss)


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

    deepspeed.init_distributed(timeout=timedelta(minutes=300))


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


def gather(item, device):
    t = torch.tensor(item, device=device)
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return gathered
    

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
    
    
    if args.score_small:
        model = setup_model(args, ds_config, device, set_optim=args.do_train)
        avg_small_loss = score_small(args, tokenizer, model, dataset, device)
        all_avg_small_loss = gather(avg_small_loss, device)
        all_avg_small_loss = [x.item() for x in all_avg_small_loss]
        if dist.get_rank() == 0:
            print("All Small Loss", all_avg_small_loss, "Avg.", sum(all_avg_small_loss) / len(all_avg_small_loss))
            
        args.ckpt_name = "gpt2-large"
        args.model_config = "/home/lidong1/CodeRepo/icl_train/results/gpt2-large"
        model = setup_model(args, ds_config, device, set_optim=args.do_train)
        avg_small_loss = score_small(args, tokenizer, model, dataset, device)
        all_avg_small_loss = gather(avg_small_loss, device)
        all_avg_small_loss = [x.item() for x in all_avg_small_loss]
        if dist.get_rank() == 0:
            print("All Vanilla Loss", all_avg_small_loss, "Avg.", sum(all_avg_small_loss) / len(all_avg_small_loss))
    else:
        model = setup_model(args, ds_config, device, set_optim=args.do_train)
        if args.lm_ratio is not None:
            avg_lm_loss = score(args, tokenizer, model, dataset, device, "lm")
            all_avg_lm_loss = gather(avg_lm_loss, device)
            all_avg_lm_loss = [x.item() for x in all_avg_lm_loss]
            if dist.get_rank() == 0:
                print("All LM Loss", all_avg_lm_loss, "Avg.", sum(all_avg_lm_loss) / len(all_avg_lm_loss))
        else:
            if args.score_icl:
                avg_icl_loss = score(args, tokenizer, model, dataset, device, "icl")
                all_avg_icl_loss = gather(avg_icl_loss, device)
                all_avg_icl_loss = [x.item() for x in all_avg_icl_loss]
                if dist.get_rank() == 0:
                    print("All ICL Loss", all_avg_icl_loss, "Avg.", sum(all_avg_icl_loss) / len(all_avg_icl_loss))
            if args.score_zero:
                avg_zs_loss = score(args, tokenizer, model, dataset, device, "zs")
                all_avg_zs_loss = gather(avg_zs_loss, device)
                all_avg_zs_loss = [x.item() for x in all_avg_zs_loss]
                if dist.get_rank() == 0:
                    print("All ZS Loss", all_avg_zs_loss, "Avg.", sum(all_avg_zs_loss) / len(all_avg_zs_loss))


if __name__ == "__main__":
    main()