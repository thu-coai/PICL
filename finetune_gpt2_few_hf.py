import sys
if "/home/lidong1/.local/lib/python3.8/site-packages" in sys.path:
    sys.path.remove("/home/lidong1/.local/lib/python3.8/site-packages")

import time
import os
from datetime import timedelta

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

import random
import json

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AdamW, get_constant_schedule_with_warmup


from model_center import get_args

from data_utils.icl_datasets import ICLEvalDataset
from data_utils.icl_few_bag_datasets import FewBagICLTrainDataset, FewBagICLEvalDataset
from data_utils.icl_vanilla_datasets import VanillaICLTrainDataset, VanillaICLEvalDataset
from utils import MultiPromptResults, print_args
from utils import print_rank_hf as print_rank
from utils import save_rank_hf as save_rank
from tqdm import tqdm

from icl_train.data_utils.data_config import DATA_CONFIG, T0_METRICS


def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)
    
    return tokenizer


def get_model(args, device):
    model = GPT2LMHeadModel.from_pretrained(args.model_config)
    model = model.to(device)
    model = DDP(model, device_ids=[device])
    return model


def get_optimizer(args, model):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, device):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
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

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=300))


def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    init_distributed(args)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args


def prepare_dataset(args, tokenizer, rank, world_size):
    data = {}
    rng_sample = random.Random(args.seed)
    rng_order = random.Random(args.seed_order)
    if args.do_train:
        if args.icl_bag_of_inst:
            data["train"] = FewBagICLTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, args.shot, rng_sample, rng_order)
            if args.do_valid:
                data["dev"] = FewBagICLEvalDataset(args, tokenizer, args.data_dir, "validation", args.dev_num, args.dev_ratio, args.shot, rng_sample, rng_order)
        else:
            data["train"] = VanillaICLTrainDataset(args, tokenizer, args.data_dir, "train", args.train_num, args.train_ratio, args.shot, rng_sample, rng_order)
            if args.do_valid:
                data["dev"] = VanillaICLEvalDataset(args, tokenizer, args.data_dir, "validation", args.dev_num, args.dev_ratio, args.shot, rng_sample, rng_order)

        data["train"].build_icl_demos_rand(data["train"].all_data, "train", pool_caches=data["train"].get_all_cache_path())
        if args.do_valid:
            data["dev"].build_icl_demos_rand(data["train"].all_data, "validation", pool_caches=data["train"].get_all_cache_path())

    elif args.do_eval:
        if args.icl_bag_of_inst:
            data["train"] = FewBagICLTrainDataset(args, tokenizer, args.data_dir, "train", -1, args.train_ratio, args.shot, rng_sample, rng_order, build_train_idx=False)
            data["test"] = FewBagICLEvalDataset(args, tokenizer, args.data_dir, "validation", args.dev_num, args.dev_ratio, args.shot, rng_sample, rng_order)
        else:
            data["train"] = VanillaICLTrainDataset(args, tokenizer, args.data_dir, "train", -1, args.train_ratio, args.shot, rng_sample, rng_order, build_train_idx=False)
            data["test"] = VanillaICLEvalDataset(args, tokenizer, args.data_dir, "validation", args.dev_num, args.dev_ratio, args.shot, rng_sample, rng_order)

        if args.icl_share_demo:
            data["test"].build_icl_demos_share(data["train"].all_data, data["train"].label_sid_map, args.icl_balance)
        else:
            data["test"].build_icl_demos_rand(data["train"].all_data, "test", pool_caches=data["train"].get_all_cache_path())
    else:
        raise ValueError("Do train and do eval must set one")
    return data


def finetune(args, tokenizer: GPT2Tokenizer, model: DDP, optimizer: AdamW, lr_scheduler, dataset, device):
    loss_func = nn.CrossEntropyLoss()

    # print_inspect(model, '*')
    collate_fn = dataset["train"].collate

    sampler = DistributedSampler(dataset["train"], shuffle=False, drop_last=True)
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    step, global_step = 1, 1
    total_loss, total_time = 0.0, 0.0

    for epoch in range(args.epochs):
        dataset["train"].set_epoch(epoch)

        model.eval()
        for it, (model_batch, no_model_batch) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, device)

            torch.cuda.synchronize()
            st_time = time.time()

            outputs = model(**model_batch)
            
            logits = outputs.logits

            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            global_loss = loss * args.gradient_accumulation_steps
            dist.all_reduce(global_loss, dist.ReduceOp.SUM)
            global_loss = global_loss.item() / dist.get_world_size()

            grad_norm = 0
            if step % args.gradient_accumulation_steps == 0:
                print_rank("optim step")
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e} | grad_norm: {:.4f} | time: {:.3f}".format(
                    epoch,
                    step,
                    len(train_dataloader) * args.epochs,
                    global_step,
                    int(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs,
                    log_loss,
                    lr_scheduler.get_last_lr()[0],
                    grad_norm,
                    log_time,
                )
                
            print_rank(get_log(global_loss, elapsed_time))

            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval * args.gradient_accumulation_steps))
                # bmt.print_rank(log_str)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                total_loss, total_time = 0.0, 0.0
            
            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_dir_path = os.path.join(args.save, str(global_step))
                if dist.get_rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    print_rank(f"Model save to {save_dir_path}")
                    model.config.to_json_file(os.path.join(save_dir_path, "config.json"))
                    tokenizer.save_pretrained(save_dir_path)
                torch.save(model.state_dict(), os.path.join(save_dir_path, "pytorch_model.pt"))

            # Evaluation
            if args.do_valid and args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                all_eval_res = evaluate_all(args, tokenizer, model, dataset["dev"], "dev", epoch, device)
                all_eval_res.save_res(args.save, global_step)
                model.train()
                
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1


def process_loss(args, losses, mask, pos_mask, gold_labels, data_name, prompt_name, device):
    losses = losses.view(mask.size())
    losses = losses * mask
    cum_losses = torch.cumsum(losses, dim=1)
    tmp_pos_index = torch.arange(1, losses.size(1) + 1, device=device)
    preds = []
    all_option_loss = []
    min_loss, gold_loss = 0, 0
    for cum_loss, pos, gold_label in zip(cum_losses, pos_mask, gold_labels):
        # deal with the case where option numbers are not equal in a batch
        sum_loss = torch.masked_select(cum_loss, pos) # the first "True" of pos is the end of the context
        sum_loss = sum_loss - sum_loss[0]
        option_loss = torch.diff(sum_loss, dim=0)
        pos_idx = torch.masked_select(tmp_pos_index, pos)
        pos_idx = pos_idx - pos_idx[0]
        option_lens = torch.diff(pos_idx, dim=0)
        if args.norm_option_loss:
            option_loss = option_loss / option_lens
        min_option_loss, min_option_idx = torch.min(option_loss, dim=0)
        min_loss += min_option_loss.item()
        gold_loss += option_loss[gold_label.item()].item()
        preds.append(min_option_idx.item())
        all_option_loss.append(option_loss)
    
    preds = torch.tensor(preds, dtype=torch.long, device=device)
    min_loss /= len(losses)
    gold_loss /= len(losses)
    return preds, min_loss, gold_loss, all_option_loss


def get_res(idxs, preds, dataset: ICLEvalDataset):
    all_labels_str, all_preds_str = [], []
    for (_, _, sid), pred in zip(idxs, preds):
        all_labels_str.append(dataset.current_data()[sid]["target_str"])
        all_preds_str.append(dataset.current_data()[sid]["options_str"][pred])
        
    metric_names = dataset.get_metrics()
    eval_res = {}
    for metric_name in metric_names:
        post_fn = DATA_CONFIG[dataset.data_name].get_answer_post_fn(metric_name)
        all_labels_str, all_preds_str = post_fn((all_labels_str, all_preds_str))
        res = T0_METRICS[metric_name](all_labels_str, all_preds_str)
        eval_res.update(res)
    
    return eval_res, all_labels_str, all_preds_str


def evaluate(args, tokenizer, model, dataset: ICLEvalDataset, epoch, device):
    
    collate_fn = dataset.collate
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    model.eval()
    all_preds, all_idxs = [], []
    all_gold_loss = 0.0
    step = 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} {dataset.prompt_name}", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch, output_logits=True).logits
            losses = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            preds, min_loss, gold_loss, option_losses = process_loss(
                args, losses, no_model_batch["loss_mask"], no_model_batch["pos_mask"], no_model_batch["option_label"], dataset.data_name, dataset.prompt_name, device)
            all_preds.append(preds)
            all_idxs.append(no_model_batch["idxs"])
            all_gold_loss += gold_loss
            step += 1

    all_preds = torch.cat(all_preds, dim=0)
    gathered_all_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_all_preds, all_preds)
    all_preds = torch.cat(gathered_all_preds, dim=0)
    
    all_idxs = torch.cat(all_idxs, dim=0)
    gathered_all_idxs = [torch.zeros_like(all_idxs) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_all_idxs, all_idxs)
    all_idxs = torch.cat(gathered_all_idxs, dim=0)
    
    all_gold_loss = all_gold_loss / step
    eval_res, all_labels_str, all_preds_str = get_res(all_idxs, all_preds, dataset)
    return eval_res, all_gold_loss, all_labels_str, all_preds_str


def evaluate_all(args, tokenizer, model, dataset: ICLEvalDataset, split, epoch, device):
    all_eval_res = MultiPromptResults(dataset.data_prompts)
    for data_name, prompt_name in dataset.data_prompt_names:
        dataset.set_name_prompt(data_name, prompt_name)
        if len(dataset) == 0:
            log_str = f"{split} | {data_name} | {prompt_name} | Data size 0, skip"
            print_rank(log_str)
            # save_rank(log_str, os.path.join(args.save, "log.txt"))
            continue
        if args.eval_test:
            eval_res, gold_loss, all_labels_str, all_preds_str = evaluate_test(args, tokenizer, model, dataset, epoch, device)
        else:
            eval_res, gold_loss, all_labels_str, all_preds_str = evaluate(args, tokenizer, model, dataset, epoch, device)
        print_rank(f"{split} | {data_name} | {prompt_name}")
        print_rank(f"{eval_res} | {gold_loss}")
        all_eval_res.add_res(dataset.data_name, dataset.prompt_name, eval_res)
        all_eval_res.add_preds(dataset.data_name, dataset.prompt_name, (all_labels_str, all_preds_str))
        all_eval_res.add_loss(dataset.data_name, dataset.prompt_name, gold_loss)
    avg_res = all_eval_res.average(key="res")
    avg_loss = all_eval_res.average(key="loss")

    log_str = f"{split} | avg_res: {avg_res} | avg_loss: {avg_loss}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
    for data_name in all_eval_res.all_data_names():
        log_res = all_eval_res.all_res(data_name, key="res")
        avg_log_res = all_eval_res.average_per_data(data_name, key="res")
        log_losses = all_eval_res.all_res(data_name, key="loss")
        avg_log_loss = all_eval_res.average_per_data(data_name, key="loss")
        log_str = f"{split} | name: {data_name} | avg res: {avg_log_res} | avg loss: {round(avg_log_loss, 4)} | res: {log_res} | loss: {log_losses}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
    return all_eval_res


def evaluate_test(args, tokenizer, model, dataset: ICLEvalDataset, epoch, device):
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=dataset.collate_test)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    model.eval()
    all_preds, all_idxs = [], []
    all_gold_loss, all_min_loss = 0.0, 0.0
    step = 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} {dataset.prompt_name}", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch, output_logits=True).logits
            losses = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            losses = losses.view(model_batch["input_ids"].size())
            losses = torch.sum(losses * no_model_batch["loss_mask"], dim=-1)
            if args.norm_option_loss:
                losses = losses / torch.sum(no_model_batch["loss_mask"], dim=-1)
            losses = losses.view(-1, dataset.label_size)
            min_option_loss, min_option_idx = torch.min(losses, dim=-1)
            min_option_loss = torch.mean(min_option_loss, dim=0)
            gold_option_loss = torch.gather(losses, dim=1, index=no_model_batch["option_label"].unsqueeze(1))
            gold_option_loss = torch.mean(gold_option_loss.squeeze(1), dim=0)
            all_gold_loss += gold_option_loss.item()
            all_min_loss += min_option_loss.item()

            all_preds.append(min_option_idx)
            all_idxs.append(no_model_batch["idxs"])
            step += 1

    all_preds = torch.cat(all_preds, dim=0)
    gathered_all_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_all_preds, all_preds)
    all_preds = torch.cat(gathered_all_preds, dim=0)
    
    all_idxs = torch.cat(all_idxs, dim=0)
    gathered_all_idxs = [torch.zeros_like(all_idxs) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_all_idxs, all_idxs)
    all_idxs = torch.cat(gathered_all_idxs, dim=0)
    
    all_gold_loss = all_gold_loss / step
    all_min_loss = all_min_loss / step
    eval_res, all_labels_str, all_preds_str = get_res(all_idxs, all_preds, dataset)
    return eval_res, all_gold_loss, all_labels_str, all_preds_str


def main():
    args = initialize()
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
        dist.get_rank(), dist.get_world_size(),
    )
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, device)
    
    if args.do_train:
        finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device)
    
    if args.do_eval:
        all_eval_res = evaluate_all(args, tokenizer, model, dataset["test"], "test", 0, device)
        all_eval_res.save_res(args.save, -1)


if __name__ == "__main__":
    print(os.environ["CODE_BASE"])
    main()
