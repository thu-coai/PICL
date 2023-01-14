import sys
if "/home/lidong1/.local/lib/python3.8/site-packages" in sys.path:
    sys.path.remove("/home/lidong1/.local/lib/python3.8/site-packages")

import time
import os

import torch
import random
import json

import bmtrain as bmt
from bmtrain.optim import AdamOffloadOptimizer, AdamOptimizer

from model_center import get_args
from model_center.model import GPT2ICL
from model_center.tokenizer import GPT2Tokenizer
from model_center.dataset import DistributedDataLoader
from data_utils.icl_datasets import ICLEvalDataset
from data_utils.icl_few_bag_datasets import FewBagICLTrainDataset, FewBagICLEvalDataset
from data_utils.icl_vanilla_datasets import VanillaICLTrainDataset, VanillaICLEvalDataset
from utils import MultiPromptResults, set_random_seed, print_args
from utils import save_rank_bmt as save_rank
from tqdm import tqdm

from icl_train.data_utils.data_config import DATA_CONFIG, T0_METRICS

def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    model = GPT2ICL.from_pretrained(args.model_config)
    return model

def get_optimizer(args, model):
    optimizer = AdamOptimizer(model.parameters(),
                                        weight_decay=args.weight_decay,
                                        scale=args.loss_scale)
    # optimizer = AdamOffloadOptimizer(model.parameters(), 
    #                                            weight_decay=args.weight_decay, 
    #                                            scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    if args.lr_decay_style == "noam":
        lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "constant":
        lr_scheduler = bmt.lr_scheduler.NoDecay(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = -1,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "exponential":
        lr_scheduler = bmt.lr_scheduler.Exponential(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = bmt.lr_scheduler.Cosine(optimizer, 
                                            start_lr = args.lr,
                                            warmup_iter = args.warmup_iters, 
                                            end_iter = args.lr_decay_iters,
                                            num_iter = args.start_step)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args):
    # get the model
    model = get_model(args)
    bmt.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    # get the memory usage
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return model, optimizer, lr_scheduler


def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 100)
    set_random_seed(args.seed)
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


def finetune(args, tokenizer: GPT2Tokenizer, model: GPT2ICL, optimizer: AdamOptimizer, lr_scheduler, dataset, device):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    # print_inspect(model, '*')
    collate_fn = dataset["train"].collate

    train_dataloader = DistributedDataLoader(
        dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    step, global_step = 1, 1
    total_loss, total_time = 0.0, 0.0

    for epoch in range(args.epochs):
        dataset["train"].set_epoch(epoch)

        model.train()
        for it, (model_batch, no_model_batch) in enumerate(train_dataloader):
            dataset["train"].move_to_device(model_batch, no_model_batch, device)

            torch.cuda.synchronize()
            st_time = time.time()

            outputs = model(**model_batch, output_logits=True)
            
            logits = outputs.logits

            loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            global_loss = bmt.distributed.all_reduce(loss, "avg").item()

            loss = loss / args.gradient_accumulation_steps
            loss = optimizer.loss_scale(loss)
            loss.backward()
            
            grad_norm = 0
            if step % args.gradient_accumulation_steps == 0:
                bmt.print_rank("optim step")
                grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale, norm_type = 2)
                bmt.optim_step(optimizer, lr_scheduler)
                optimizer.zero_grad()

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} | time: {:.3f}".format(
                    epoch,
                    step,
                    len(train_dataloader) * args.epochs,
                    global_step,
                    int(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs,
                    log_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    grad_norm,
                    log_time,
                )
                
            bmt.print_rank(get_log(global_loss, elapsed_time))

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
                if bmt.rank() == 0:
                    os.makedirs(save_dir_path, exist_ok=True)
                    bmt.print_rank(f"Model save to {save_dir_path}")
                    model.config.to_json_file(os.path.join(save_dir_path, "config.json"))
                    tokenizer.save_pretrained(save_dir_path)
                bmt.save(model, os.path.join(save_dir_path, "pytorch_model.pt"))

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
    
    dataloader = DistributedDataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")
    model.eval()
    all_preds, all_idxs = [], []
    all_gold_loss = 0.0
    step = 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} {dataset.prompt_name}", disable=(bmt.rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            logits = model(**model_batch, output_logits=True).logits
            losses = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            preds, min_loss, gold_loss, option_losses = process_loss(
                args, losses, no_model_batch["loss_mask"], no_model_batch["pos_mask"], no_model_batch["option_label"], dataset.data_name, dataset.prompt_name, device)
            all_preds.append(preds)
            all_idxs.append(no_model_batch["idxs"])
            all_gold_loss += gold_loss
            step += 1

    all_preds = bmt.gather_result(torch.cat(all_preds, dim=0)).cpu().tolist()
    all_idxs = bmt.gather_result(torch.cat(all_idxs, dim=0)).cpu().tolist()
    all_gold_loss = all_gold_loss / step
    eval_res, all_labels_str, all_preds_str = get_res(all_idxs, all_preds, dataset)
    return eval_res, all_gold_loss, all_labels_str, all_preds_str


def evaluate_all(args, tokenizer, model, dataset: ICLEvalDataset, split, epoch, device):
    all_eval_res = MultiPromptResults(dataset.data_prompts)
    for data_name, prompt_name in dataset.data_prompt_names:
        dataset.set_name_prompt(data_name, prompt_name)
        if len(dataset) == 0:
            log_str = f"{split} | {data_name} | {prompt_name} | Data size 0, skip"
            bmt.print_rank(log_str)
            # save_rank(log_str, os.path.join(args.save, "log.txt"))
            continue
        if args.eval_test:
            eval_res, gold_loss, all_labels_str, all_preds_str = evaluate_test(args, tokenizer, model, dataset, epoch, device)
        else:
            eval_res, gold_loss, all_labels_str, all_preds_str = evaluate(args, tokenizer, model, dataset, epoch, device)
        bmt.print_rank(f"{split} | {data_name} | {prompt_name}")
        bmt.print_rank(f"{eval_res} | {gold_loss}")
        all_eval_res.add_res(dataset.data_name, dataset.prompt_name, eval_res)
        all_eval_res.add_preds(dataset.data_name, dataset.prompt_name, (all_labels_str, all_preds_str))
        all_eval_res.add_loss(dataset.data_name, dataset.prompt_name, gold_loss)
    avg_res = all_eval_res.average(key="res")
    avg_loss = all_eval_res.average(key="loss")

    log_str = f"{split} | avg_res: {avg_res} | avg_loss: {avg_loss}"
    bmt.print_rank(log_str)
    save_rank(log_str, os.path.join(args.save, "log.txt"))
    for data_name in all_eval_res.all_data_names():
        log_res = all_eval_res.all_res(data_name, key="res")
        avg_log_res = all_eval_res.average_per_data(data_name, key="res")
        log_losses = all_eval_res.all_res(data_name, key="loss")
        avg_log_loss = all_eval_res.average_per_data(data_name, key="loss")
        log_str = f"{split} | name: {data_name} | avg res: {avg_log_res} | avg loss: {round(avg_log_loss, 4)} | res: {log_res} | loss: {log_losses}"
        bmt.print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
    return all_eval_res


def evaluate_test(args, tokenizer, model, dataset: ICLEvalDataset, epoch, device):
    dataloader = DistributedDataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_test)
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")
    model.eval()
    all_preds, all_idxs = [], []
    all_gold_loss, all_min_loss = 0.0, 0.0
    step = 0
    with torch.no_grad():
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} {dataset.prompt_name}", disable=(bmt.rank() != 0))):
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

    all_preds = bmt.gather_result(torch.cat(all_preds, dim=0)).cpu().tolist()
    all_idxs = bmt.gather_result(torch.cat(all_idxs, dim=0)).cpu().tolist()
    all_gold_loss = all_gold_loss / step
    all_min_loss = all_min_loss / step
    eval_res, all_labels_str, all_preds_str = get_res(all_idxs, all_preds, dataset)
    return eval_res, all_gold_loss, all_labels_str, all_preds_str


# def evaluate_lm(args, tokenizer, model, dataset: ICLPtEvalDataset, epoch, device):
#     dataloader = DistributedDataLoader(
#         dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=dataset.collate_lm)
#     loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")
#     model.eval()
#     all_preds, all_idxs = [], []
#     all_gold_loss = 0.0
#     step = 0
#     with torch.no_grad():
#         for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} {dataset.prompt_name}", disable=(bmt.rank() != 0))):
#             # torch.save(model_batch, "model_batch_dev.pt")
#             # torch.save(no_model_batch, "no_model_batch_dev.pt")
#             # exit(0)
#             dataset.move_to_device(model_batch, no_model_batch, device)
#             logits = model(**model_batch, output_logits=True).logits
#             loss = loss_func(logits.view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
#             loss = loss.view(no_model_batch["label"].size())
#             # torch.save(model_batch["input_ids"][0], "input_ids_lm.pt")
#             # print(tokenizer.decode(model_batch["input_ids"][0]))
#             print(torch.masked_select(loss[0], no_model_batch["loss_mask"][0].bool()))
#             print(logits.size())
#             print(logits[0])
#             return
#             # print(loss)


def main():
    args = initialize()
    
    if bmt.rank() == 0:
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
        bmt.rank(), bmt.world_size(),
    )
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    
    if args.do_train:
        finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device)
    
    if args.do_eval:
        all_eval_res = evaluate_all(args, tokenizer, model, dataset["test"], "test", 0, device)
        all_eval_res.save_res(args.save, -1)

if __name__ == "__main__":
    main()
