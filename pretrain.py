import sys

import time
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
# from apex.optimizers import FusedAdam as Adam
from torch.optim import AdamW
import deepspeed

import random
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

from arguments import get_picl_pretrain_args

from data_utils.pretrain_datasets import ICLPretrainDataset
from utils import get_optimizer_params, set_random_seed, print_args
from utils import print_rank, save_rank
from tqdm import tqdm


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    return tokenizer


def get_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def get_optimizer(args, model):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, DDP):
        model = model.module

    param_groups = get_optimizer_params(args, model)

    # Use FusedAdam.
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.lr_decay_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
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
    args = get_picl_pretrain_args()
    # init bmt 
    init_distributed(args)
    set_random_seed(args.seed)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, rank, world_size):
    data = {}
    rng_sample = random.Random(args.seed)

    def _get_dataset(split, mode):
        num = args.picl_train_num if split == "train" else args.picl_valid_num
        lm_num = args.lm_train_num if split == "train" else args.lm_valid_num
        _dataset = ICLPretrainDataset(
                args,
                tokenizer,
                path_icl=args.picl_data_dir,
                split=split,
                num=num,
                lm_num=lm_num,
                shot=args.shot,
                mode=mode,
                path_lm=args.lm_data_dir,
                path_icl_idx=args.picl_idx_data_dir,
                rng_sample=rng_sample)
        return _dataset
    
    if args.do_train:
        if args.pretrain_type in ["mixed", "icl"]:
            data["train_icl"] = _get_dataset("train", "icl")
            data["dev_icl"] = _get_dataset("valid", "icl")

        if args.pretrain_type in ["mixed", "lm"]:
            data["train_lm"] = _get_dataset("train", "lm")
            data["dev_lm"] = _get_dataset("valid", "lm")

        if args.train_iters == -1:
            if args.pretrain_type in ["mixed", "icl"]:
                args.train_iters = int(len(data["train_icl"]) / (args.batch_size * dist.get_world_size() * args.gradient_accumulation_steps))
            else:
                args.train_iters = int(len(data["train_lm"]) / (args.batch_size * dist.get_world_size() * args.gradient_accumulation_steps))
                
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters * args.epochs
        print_rank("Train iters per epoch", args.train_iters)
        print_rank("lr_decay_iters", args.lr_decay_iters)
    
    elif args.do_eval:
        if args.pretrain_type in ["mixed", "icl"]:
            data["test_icl"] = _get_dataset("valid", "icl")
        if args.pretrain_type in ["mixed", "lm"]:
            data["test_lm"] = _get_dataset("valid", "lm")
    else:
        raise ValueError("Do train and do eval must set one")
    return data


def pretrain(args, tokenizer: AutoTokenizer, model: deepspeed.DeepSpeedEngine, optimizer: AdamW, lr_scheduler, dataset, device):
    print_rank("Start Pre-training")
    loss_func = nn.CrossEntropyLoss()

    # print_inspect(model, '*')
    if args.pretrain_type in ["mixed", "icl"]:
        collate_fn = dataset["train_icl"].collate
        sampler_icl = DistributedSampler(dataset["train_icl"], shuffle=True, drop_last=True)
        train_icl_dataloader = DataLoader(
            dataset['train_icl'], sampler=sampler_icl, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    
    if args.pretrain_type in ["mixed", "lm"]:
        collate_fn = dataset["train_lm"].collate
        sampler_lm = DistributedSampler(dataset["train_lm"], shuffle=True, drop_last=True)
        train_lm_dataloader = DataLoader(
            dataset['train_lm'], sampler=sampler_lm, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    if args.pretrain_type in ["mixed", "icl"]:
        train_dataloader = train_icl_dataloader
        sampler = sampler_icl
        train_dataset = dataset["train_icl"]
        if args.pretrain_type == "mixed":
            train_lm_dataloader = iter(train_lm_dataloader)
            sampler_lm.set_epoch(0)
    else:
        train_dataloader = train_lm_dataloader
        sampler = sampler_lm
        train_dataset = dataset["train_lm"]

    step, global_step = 1, 1
    total_loss, total_loss_lm, total_time = 0.0, 0.0, 0.0

    lm_epoch = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch) in enumerate(train_dataloader):
            
            if it == 0:
                train_dataset.show_example()
                torch.save(model_batch, os.path.join(args.save, f"model_batch_{dist.get_rank()}.pt"))
                torch.save(no_model_batch, os.path.join(args.save, f"no_model_batch_{dist.get_rank()}.pt"))
            train_dataset.move_to_device(model_batch, no_model_batch, device)

            if args.pretrain_type == "mixed":
                try:
                    lm_model_batch, lm_no_model_batch = next(train_lm_dataloader)
                except StopIteration:
                    train_lm_dataloader = iter(train_lm_dataloader)
                    lm_epoch += 1
                    sampler_lm.set_epoch(lm_epoch)
                    lm_model_batch, lm_no_model_batch = next(train_lm_dataloader)
                
                if it == 0:
                    dataset["train_lm"].show_example()
                    torch.save(lm_model_batch, os.path.join(args.save, f"lm_model_batch_{dist.get_rank()}.pt"))
                    torch.save(lm_no_model_batch, os.path.join(args.save, f"lm_no_model_batch_{dist.get_rank()}.pt"))
                dataset["train_lm"].move_to_device(lm_model_batch, lm_no_model_batch, device)

            torch.cuda.synchronize()
            st_time = time.time()
            
            outputs = model(**model_batch, use_cache=False)
            logits = outputs.logits
            loss = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            
            if args.pretrain_type == "mixed":
                outputs_lm = model(**lm_model_batch, use_cache=False)
                logits_lm = outputs_lm.logits
                loss_lm = loss_func(logits_lm.float().view(-1, logits_lm.shape[-1]), lm_no_model_batch["label"].view(-1))
                lm_ratio = args.lm_ratio
                loss = (loss + lm_ratio * loss_lm) / (1 + lm_ratio)
            
            model.backward(loss)
            model.step()
            
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            global_loss = loss.item() / dist.get_world_size()
            global_loss_lm = 0
            if args.pretrain_type == "mixed":
                dist.all_reduce(loss_lm, dist.ReduceOp.SUM)
                global_loss_lm = loss_lm.item() / dist.get_world_size()

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            if args.pretrain_type == "mixed":
                total_loss_lm += global_loss_lm
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_lm_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | lm_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    len(train_dataloader) * args.epochs,
                    global_step,
                    int(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs,
                    log_loss,
                    log_lm_loss,
                    lr_scheduler.get_last_lr()[0],
                    optimizer.cur_scale,
                    elapsed_time,
                    log_time,
                )
            
            # if step % (args.gradient_accumulation_steps // 4) == 0:
            print_rank(get_log(global_loss, global_loss_lm, 0))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_loss_lm / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_time = 0.0, 0.0
                total_loss_lm = 0.0
            
            # Checkpointing
            if args.save and (global_step == 200) or (args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0):
                save_dir_path = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    print_rank(f"Model save to {save_dir_path}")
                    model.module.config.to_json_file(os.path.join(save_dir_path, "config.json"))
                    tokenizer.save_pretrained(save_dir_path)
                    torch.save(model.module.state_dict(), os.path.join(save_dir_path, "pytorch_model.bin"))

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                if args.pretrain_type in ["mixed", "icl"]:
                    dev_loss = evaluate(args, tokenizer, model, dataset["dev_icl"], "dev", epoch, "icl", device)
                if args.pretrain_type in ["mixed", "lm"]:
                    dev_loss = evaluate(args, tokenizer, model, dataset["dev_lm"], "dev", epoch, "lm", device)
                    
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1


def evaluate(args, tokenizer, model, dataset: ICLPretrainDataset, split, epoch, mode, device):
    
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
            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch).logits
            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            all_loss += loss.item()
            step += 1
    
    avg_loss = all_loss / step
    
    log_str = f"{mode} | {split} | avg_loss: {avg_loss}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
    
    return avg_loss


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
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    if args.do_train:
        pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset, device)
    
    if args.do_eval:
        if args.pretrain_type in ["mixed", "icl"]:
            loss = evaluate(args, tokenizer, model, dataset["test_icl"], "test", 0, "icl", device)
        if args.pretrain_type in ["mixed", "lm"]:
            loss = evaluate(args, tokenizer, model, dataset["test_lm"], "test", 0, "lm", device)


if __name__ == "__main__":
    main()
