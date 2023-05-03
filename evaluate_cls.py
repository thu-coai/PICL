import time
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
from datetime import timedelta

import random
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

from arguments import get_picl_eval_args

from data_utils.train_dataset import ICLTrainDataset
from data_utils.evaluation_datasets import ICLEvalCLSDataset 
from utils import MultiPromptResults, set_random_seed, print_args, print_rank, save_rank
from tqdm import tqdm

from data_utils.data_config import DATA_CONFIG, T0_METRICS


torch.set_num_threads(4)


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    return tokenizer


def get_model(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler

    optimizer, lr_scheduler = None, None
        
    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        lr_scheduler=lr_scheduler,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())
    return model


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
    args = get_picl_eval_args()
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
    rng_order = random.Random(args.seed_order)

    data["train"] = ICLTrainDataset(args, tokenizer, args.data_dir, "train", -1, args.train_ratio, args.shot, rng_sample, rng_order, build_train_idx=False)
    data["test"] = ICLEvalCLSDataset(args, tokenizer, args.data_dir, "validation", args.dev_num, args.dev_ratio, args.shot, rng_sample, rng_order)

    if args.icl_share_demo:
        data["test"].build_icl_demos_share(data["train"].all_data, data["train"].label_sid_map, args.icl_balance)
    else:
        data["test"].build_icl_demos_rand(data["train"].all_data, "test", pool_caches=data["train"].get_all_cache_path())

    return data


def process_loss(args, losses, mask, pos_mask, gold_labels, input_lens, data_name, prompt_name, device):
    losses = losses.view(mask.size())
    losses = losses * mask

    cum_losses = torch.cumsum(losses, dim=1)
    tmp_pos_index = torch.arange(1, losses.size(1) + 1, device=device)
    preds = []
    all_option_loss = []
    min_loss, gold_loss, gold_tot_loss = 0, 0, 0
    for cum_loss, pos, gold_label, input_len in zip(cum_losses, pos_mask, gold_labels, input_lens):
        # deal with the case where option numbers are not equal in a batch
        sum_loss = torch.masked_select(cum_loss, pos) # the first "True" of pos is the end of the context
        sum_prefix_loss = sum_loss[0]
        sum_loss = sum_loss - sum_loss[0]
        option_loss = torch.diff(sum_loss, dim=0)
        pos_idx = torch.masked_select(tmp_pos_index, pos)
        pos_idx = pos_idx - pos_idx[0]
        option_lens = torch.diff(pos_idx, dim=0)
        normed_option_loss = option_loss / option_lens
        if args.norm_option_loss:
            option_loss = normed_option_loss
        min_option_loss, min_option_idx = torch.min(option_loss, dim=0)
        min_loss += min_option_loss.item()
        gold_loss += normed_option_loss[gold_label.item()].item()
        gold_tot_loss += ((sum_prefix_loss + option_loss[gold_label.item()]) / (input_len + option_lens[gold_label.item()])).item()
        preds.append(min_option_idx.item())
        all_option_loss.append(option_loss)
    
    preds = torch.tensor(preds, dtype=torch.long, device=device)
    min_loss /= len(losses)
    gold_loss /= len(losses)
    gold_tot_loss /= len(losses)
    return preds, min_loss, gold_loss, gold_tot_loss, all_option_loss


def get_res(idxs, preds, dataset: ICLEvalCLSDataset):
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


def evaluate(args, tokenizer, model, dataset: ICLEvalCLSDataset, epoch, device):
    
    collate_fn = dataset.collate
    
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)
    loss_func = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    model.eval()
    all_preds, all_idxs = [], []
    all_gold_loss = 0.0
    all_gold_tot_loss = 0.0
    step = 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} {dataset.prompt_name}", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            outputs = model(**model_batch)
            logits = outputs.logits
            losses = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            preds, min_loss, gold_loss, gold_tot_loss, option_losses = process_loss(
                args, losses, no_model_batch["loss_mask"], no_model_batch["pos_mask"], no_model_batch["option_label"], no_model_batch["input_lens"], dataset.data_name, dataset.prompt_name, device)

            all_preds.append(preds)
            all_idxs.append(no_model_batch["idxs"])
            all_gold_loss += gold_loss
            all_gold_tot_loss += gold_tot_loss
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
    all_gold_tot_loss = all_gold_tot_loss / step
    eval_res, all_labels_str, all_preds_str = get_res(all_idxs, all_preds, dataset)
    return eval_res, all_gold_loss, all_gold_tot_loss, all_labels_str, all_preds_str


def evaluate_all(args, tokenizer, model, dataset: ICLEvalCLSDataset, split, epoch, device):
    all_eval_res = MultiPromptResults(dataset.data_prompts)
    for data_name, prompt_name in dataset.data_prompt_names:
        dataset.set_name_prompt(data_name, prompt_name)
        if len(dataset) == 0:
            log_str = f"{split} | {data_name} | {prompt_name} | Data size 0, skip"
            print_rank(log_str)
            # save_rank(log_str, os.path.join(args.save, "log.txt"))
            continue

        eval_res, gold_loss, gold_tot_loss, all_labels_str, all_preds_str = evaluate(args, tokenizer, model, dataset, epoch, device)
                
        print_rank(f"{split} | {data_name} | {prompt_name}")
        print_rank(f"{eval_res} | {gold_loss}")
        all_eval_res.add_res(dataset.data_name, dataset.prompt_name, eval_res)
        all_eval_res.add_preds(dataset.data_name, dataset.prompt_name, (all_labels_str, all_preds_str))
        all_eval_res.add_loss(dataset.data_name, dataset.prompt_name, gold_loss)
        all_eval_res.add_tot_loss(dataset.data_name, dataset.prompt_name, gold_tot_loss)
    avg_res = all_eval_res.average(key="res")
    avg_loss = all_eval_res.average(key="loss")
    avg_tot_loss = all_eval_res.average(key="tot_loss")

    log_str = f"{split} | avg_res: {avg_res} | avg_loss: {avg_loss} | avg_tot_loss: {avg_tot_loss}"
    print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
    for data_name in all_eval_res.all_data_names():
        log_res = all_eval_res.all_res(data_name, key="res")
        avg_log_res = all_eval_res.average_per_data(data_name, key="res")
        log_losses = all_eval_res.all_res(data_name, key="loss")
        avg_log_loss = all_eval_res.average_per_data(data_name, key="loss")
        log_tot_losses = all_eval_res.all_res(data_name, key="tot_loss")
        avg_log_tot_loss = all_eval_res.average_per_data(data_name, key="tot_loss")
        log_str = f"{split} | name: {data_name} | avg res: {avg_log_res} | avg loss: {round(avg_log_loss, 4)} | avg tot loss: {round(avg_log_tot_loss, 4)} | res: {log_res} | loss: {log_losses} | tot loss: {log_tot_losses}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
    return all_eval_res


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

    ds_config["zero_optimization"]["stage"] = 0
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    dataset = prepare_dataset(
        args,
        tokenizer,
        dist.get_rank(), dist.get_world_size(),
    )
    
    model = setup_model_and_optimizer(args, ds_config, device)
    
    all_eval_res = evaluate_all(args, tokenizer, model, dataset["test"], "test", 0, device)
    all_eval_res.save_res(args.save, -1)


if __name__ == "__main__":
    main()
