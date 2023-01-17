import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import time
from datetime import timedelta
import deepspeed
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

from arguments import get_picl_eval_args

from utils import get_rank, set_random_seed, print_args, print_rank, save_rank
from tqdm import tqdm
import torch.distributed as dist

from data_utils.sni_evaluation import compute_score
from data_utils.evaluation_datasets import ICLEvalSNIDataset


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


def evaluate_gen(args, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, dataset: ICLEvalSNIDataset, epoch, device):
    
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
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} YN No", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
                        
            outputs = model(**model_batch)
            logits = outputs.logits
            losses = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            losses = losses.view(*no_model_batch["loss_mask"].size())
            gold_loss = torch.mean(torch.sum((losses * no_model_batch["loss_mask"]), dim=-1) / torch.sum(no_model_batch["loss_mask"], dim=-1), dim=0)
            tot_loss_mask = (no_model_batch["label"] != -100)
            gold_tot_loss = torch.mean(torch.sum((losses * tot_loss_mask), dim=-1) / torch.sum(tot_loss_mask, dim=-1), dim=0)
            
            preds = model.generate(
                input_ids=no_model_batch["gen_input_ids"],
                attention_mask=no_model_batch["gen_attention_mask"],
                position_ids=no_model_batch["gen_position_ids"],
                max_length=args.max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                eos_token_id=198)
            
            preds = preds[:, no_model_batch["gen_input_ids"].size(1):]
            buffer = torch.ones(len(preds), args.max_length, dtype=torch.long, device=preds.device) * tokenizer.eos_token_id
            buffer[:, :preds.size(1)] = preds

            all_preds.append(buffer)
            all_idxs.append(no_model_batch["idxs"])
            all_gold_loss += gold_loss.item()
            all_gold_tot_loss += gold_tot_loss.item()
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
    all_preds_str = tokenizer.batch_decode(all_preds, skip_special_tokens=True)
    all_preds_str = [pred.strip() for pred in all_preds_str]
    all_ids = [dataset.cur_data[sid]["id"] for sid in all_idxs.cpu().tolist()]
    # eval_res, all_labels_str, all_preds_str = get_res_gen(all_idxs, all_preds, dataset)
    return all_gold_loss, all_gold_tot_loss, all_preds_str, all_ids


def process_loss(args, losses, mask, pos_mask, gold_labels, input_lens, data_name, device):
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


def evaluate_yn(args, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, dataset: ICLEvalSNIDataset, epoch, device):
    collate_fn = dataset.collate_yn
    
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
        for it, (model_batch, no_model_batch) in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset.data_name} YN Yes", disable=(dist.get_rank() != 0))):
            dataset.move_to_device(model_batch, no_model_batch, device)
            outputs = model(**model_batch)
            logits = outputs.logits
            losses = loss_func(logits.float().view(-1, logits.shape[-1]), no_model_batch["label"].view(-1))
            preds, min_loss, gold_loss, gold_tot_loss, option_losses = process_loss(
                args, losses, no_model_batch["loss_mask"], no_model_batch["pos_mask"], no_model_batch["yn_label"], no_model_batch["input_lens"], dataset.data_name, device)

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

    all_preds_str = [dataset.cur_yn_str[p].strip() for p in all_preds]
    all_ids = [dataset.cur_data[sid]["id"] for sid in all_idxs.cpu().tolist()]
    
    return all_gold_loss, all_gold_tot_loss, all_preds_str, all_ids


def evaluate_all(args, tokenizer, model, dataset: ICLEvalSNIDataset, split, epoch, device):
    for data_name in dataset.data_names:
        set_random_seed(args.seed)
        dataset.set_name(data_name)
        if len(dataset) == 0:
            log_str = f"{split} | {data_name} | Data size 0, skip"
            print_rank(log_str)
            # save_rank(log_str, os.path.join(args.save, "log.txt"))
            continue

        if dataset.is_yn():
            gold_loss, gold_tot_loss, all_preds_str, all_ids = evaluate_yn(args, tokenizer, model, dataset, epoch, device)
        else:
            gold_loss, gold_tot_loss, all_preds_str, all_ids = evaluate_gen(args, tokenizer, model, dataset, epoch, device)
        
        log_str = f"{split} | {data_name} | gold loss: {gold_loss} | gold tot loss: {gold_tot_loss}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))
        save_res(args.save, data_name, -1, all_ids, all_preds_str)

 
def save_res(save_dir, data_name, step, ids, preds):
    if get_rank() == 0:
        save_dir = os.path.join(save_dir, "preds", str(step))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{data_name}.jsonl"), "w") as f:
            for i, p in zip(ids, preds):
                f.write(json.dumps({
                    "id": i,
                    "prediction": p
                }) + "\n")
    

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
    
    dataset = ICLEvalSNIDataset(args, tokenizer, args.data_dir)
    
    model = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
        
    evaluate_all(args, tokenizer, model, dataset, "test", 0, device)
    compute_score(args.sni_ref_file, args.save)


if __name__ == "__main__":
    print(os.environ["CODE_BASE"])
    main()
