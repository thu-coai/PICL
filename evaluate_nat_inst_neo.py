import sys
if "/home/lidong1/.local/lib/python3.8/site-packages" in sys.path:
    sys.path.remove("/home/lidong1/.local/lib/python3.8/site-packages")

import os
import pickle
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import random
import time
from datetime import timedelta
import deepspeed
import json
import re

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from model_center import get_args

from torch.utils.data import Dataset
from utils import print_rank_hf as print_rank
from utils import save_rank_hf as save_rank
from utils import get_rank, barrier, set_random_seed, print_args
from tqdm import tqdm
from ni_tasks import NI_TASKS
import torch.distributed as dist

from evaluation_ni import compute_score


yn_data = [
    "task202_mnli_contradiction_classification",
    "task1344_glue_entailment_classification",
    "task1439_doqa_cooking_isanswerable",
    "task242_tweetqa_classification",
    "task1442_doqa_movies_isanswerable",
    "task233_iirc_link_exists_classification",
    "task520_aquamuse_answer_given_in_passage",
    "task050_multirc_answerability",
    "task879_schema_guided_dstc8_classification",
    "task200_mnli_entailment_classification",
    "task623_ohsumed_yes_no_answer_generation",
    "task020_mctaco_span_based_question",
    "task226_english_language_answer_relevance_classification",
    "task199_mnli_classification",
    "task642_esnli_classification",
    "task1388_cb_entailment",
    "task1155_bard_analogical_reasoning_trash_or_treasure",
    "task827_copa_commonsense_reasoning",
    "task1624_disfl_qa_question_yesno_classification",
    "task738_perspectrum_classification",
]


class NIEvalDataset(Dataset):
    def __init__(self, args, tokenizer, path):
        self.data_name = None
        self.data_names = NI_TASKS
        self.path = path
        self.tokenizer = tokenizer
        self.args = args
        self.pad_id = tokenizer.eos_token_id
        self.max_length = args.max_length
        self.max_length_per_sample = args.max_length_per_sample
        
        self.all_data = self._load_data(self.data_names)
        self.all_max_length, self.all_gen_max_length = self._get_valid_lengths(self.all_data)
        self.all_yn, self.all_yn_str = self._get_yn(self.all_data)

        print(self.all_max_length)
        print(self.all_gen_max_length)
        print(self.all_yn)
        print(self.all_yn_str)

    def set_name(self, data_name):
        self.data_name = data_name

    @property
    def cur_name(self):
        return self.data_name

    @property
    def cur_data(self):
        return self.all_data[self.data_name]

    @property
    def cur_yn_str(self):
        return self.all_yn_str[self.data_name]

    def is_yn(self):
        return self.data_name in yn_data

    def _get_valid_lengths(self, all_data):
        all_max_length = {}
        all_gen_max_length = {}
        for dn in all_data:
            all_max_length[dn] = max([len(x["context_ids"]) + len(x["target_ids"]) for x in all_data[dn]])
            all_gen_max_length[dn] = max([len(x["context_ids"]) for x in all_data[dn]])
 
        return all_max_length, all_gen_max_length

    def _get_yn(self, all_data):
        all_yn = {}
        all_yn_str = {}
        for dn in all_data:
            if dn in yn_data:
                yn = list(set([tuple(x["target_ids"]) for x in all_data[dn]]))
                yn = [list(x) for x in yn]
                random.shuffle(yn)
                all_yn[dn] = yn
                all_yn_str[dn] = [self.tokenizer.decode(x).strip() for x in yn]
        return all_yn, all_yn_str

    def _load_from_cache(self, data_name):
        cache_path = self._get_cache_path(data_name)
        print_rank("load from", cache_path)
        data = None
        if os.path.exists(os.path.join(cache_path, "data.pkl")):
            with open(os.path.join(cache_path, "data.pkl"), "rb") as f:
                data = pickle.load(f)
        print_rank("load end")
        return data

    def _save_to_cache(self, data_name, data):
        cache_path = self._get_cache_path(data_name)
        print_rank("save to", cache_path)
        os.makedirs(cache_path, exist_ok=True)
        with open(os.path.join(cache_path, "data.pkl"), "wb") as f:
            pickle.dump(data, f)
        print_rank("save end")

    def _get_cache_path(self, data_name):
        data_dir = os.path.join(self.args.base_path, self.path, data_name)
        r2s = "r2s" if self.args.replace_return_with_space else ""
        trim = "trim" if self.args.trim else ""
        
        cache_path = os.path.join(data_dir, f"icl_new_cache/{r2s}/{trim}")
        return cache_path

    def _load_data(self, data_names):
        all_data = {}
            
        for dn in data_names:
            if self.args.force_process or \
                not os.path.exists(os.path.join(self._get_cache_path(dn), "data.pkl")):
                if get_rank() == 0:
                    data = self._process_data(dn)
                    self._save_to_cache(dn, data)
            barrier()

            data = self._load_from_cache(dn)
            all_data[dn] = data

        return all_data

    def _trunc_definition(self, definition):
        if self.args.trim:
            definition = re.sub("\n+", "\n", definition)

        if self.args.replace_return_with_space:
            definition = definition.replace("\n", " ")
            
        return definition

    def _trunc_data(self, inp, out):
        if self.args.trim:
            inp = re.sub("\n+", "\n", inp)
            out = re.sub("\n+", "\n", out)

        if self.args.replace_return_with_space:
            inp = inp.replace("\n", " ")
            out = out.replace("\n", " ")
        
        inp_ids = self.tokenizer.encode(inp)
        out_ids = self.tokenizer.encode(out)
        
        if len(inp_ids) + len(out_ids) > self.max_length_per_sample:
            inp_ids = inp_ids[-(self.max_length_per_sample-len(out_ids)):]
            
        inp = self.tokenizer.decode(inp_ids)
        
        return inp, out

    def _process_data(self, dn):
        with open(os.path.join(self.path, f"{dn}.json")) as f:
            data = json.load(f)
        
        processed_data = []
        
        inp_1, out_1 = self._trunc_data(data["Positive Examples"][0]["input"], data["Positive Examples"][0]["output"])
        inp_2, out_2 = self._trunc_data(data["Positive Examples"][1]["input"], data["Positive Examples"][1]["output"])
        if len(data["Positive Examples"]) > 2:
            inp_3, out_3 = self._trunc_data(data["Positive Examples"][2]["input"], data["Positive Examples"][2]["output"])
        else:
            inp_3, out_3 = None, None

        definition = self._trunc_definition(data["Definition"][0])
        
        sid = 0
        for d in data["Instances"][:100]:
            inp, out = self._trunc_data(d["input"], d["output"][0])
                        
            input_template = "Definition: {definition}\n" + \
                             "Positive Example 1- input: {input_1} output: {output_1}\n" + \
                             "Positive Example 2- input: {input_2} output: {output_2}\n" + \
                             ("Positive Example 3- input: {input_3} output: {output_3}\n" if inp_3 is not None else "") + \
                             "Now complete the following example- input: {input} output:"
            output_template = " {output}\n"
            
            context = input_template.format(
                definition=definition,
                input_1=inp_1, output_1=out_1,
                input_2=inp_2, output_2=out_2,
                input_3=inp_3, output_3=out_3,
                input=inp
            )
            
            target = output_template.format(
                output=out
            )
            
            context_ids = self.tokenizer.encode(context)
            target_ids = self.tokenizer.encode(target)
                        
            processed_data.append({
                "sid": sid,
                "id": d["id"],
                "context_ids": context_ids,
                "target_ids": target_ids,
            })

            sid += 1
        
        return processed_data
    
    def __len__(self):
        return len(self.all_data[self.data_name])

    def __getitem__(self, index):
        return self.all_data[self.data_name][index]
    
    def collate(self, samples):
        bs = len(samples)
        max_length = self.all_max_length[self.data_name]
        gen_max_length = self.all_gen_max_length[self.data_name]
        model_batch = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "position_ids": torch.zeros(bs, max_length, dtype=torch.long),
            "attention_mask": torch.zeros(bs, max_length, max_length, dtype=torch.long)
        }
        
        no_model_batch = {
            "idxs": torch.zeros(bs, dtype=torch.long),
            "gen_input_ids": torch.ones(bs, gen_max_length, dtype=torch.long) * self.pad_id,
            "gen_position_ids": torch.zeros(bs, gen_max_length, dtype=torch.long),
            "gen_attention_mask": torch.zeros(bs, gen_max_length, dtype=torch.long),
            "label": torch.ones(bs, max_length, dtype=torch.long),
            "loss_mask": torch.zeros(bs, max_length, dtype=torch.long)
        }
        
        for b, samp in enumerate(samples):
            input_len = len(samp["context_ids"]) + len(samp["target_ids"])
            model_batch["input_ids"][b][:input_len-1] = torch.tensor(samp["context_ids"] + samp["target_ids"][:-1], dtype=torch.long)
            model_batch["position_ids"][b][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
            model_batch["attention_mask"][b][:input_len-1, :input_len-1] = torch.tril(torch.ones(input_len-1, input_len-1))
            
            no_model_batch["idxs"][b] = samp["sid"]
            no_model_batch["label"][b][:input_len-1] = torch.tensor(samp["context_ids"][1:] + samp["target_ids"], dtype=torch.long)
            no_model_batch["loss_mask"][b][:input_len-1] = 1.0
            no_model_batch["gen_input_ids"][b][-len(samp["context_ids"]):] = torch.tensor(samp["context_ids"], dtype=torch.long)
            no_model_batch["gen_position_ids"][b][-len(samp["context_ids"]):] = torch.arange(0, len(samp["context_ids"]), dtype=torch.long)
            no_model_batch["gen_attention_mask"][b][-len(samp["context_ids"]):] = 1.0
    
        return model_batch, no_model_batch

    def collate_yn(self, samples):
        bs = len(samples)
        assert self.data_name in self.all_yn
        max_options_sizes = sum(map(len, self.all_yn[self.data_name]))
        max_length = self.all_gen_max_length[self.data_name]
        
        # max_length has contained max_options_sizes, but is large enough
        
        model_data = {
            "input_ids": torch.ones(bs, max_length + max_options_sizes, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length + max_options_sizes, max_length + max_options_sizes),
            "position_ids": torch.zeros(bs, max_length + max_options_sizes, dtype=torch.long),
        }
        no_model_data = {
            "idxs": torch.zeros(bs, dtype=torch.long),
            "yn_label": torch.ones(bs, dtype=torch.long),
            "label": torch.ones(bs, max_length + max_options_sizes, dtype=torch.long) * (-100),
            "loss_mask": torch.zeros(bs, max_length + max_options_sizes),
            "pos_mask": torch.zeros(bs, max_length + max_options_sizes, dtype=torch.bool),
            "input_lens": torch.zeros(bs, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            icl_prefix_ids = samp["context_ids"]
            input_len = len(icl_prefix_ids) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(icl_prefix_ids[:-1], dtype=torch.long) # we move the last token of the prefix to the begining of each option
            model_data["attention_mask"][i][:input_len, :input_len] = torch.tril(torch.ones(input_len, input_len))
            model_data["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            no_model_data["label"][i][:input_len] = torch.tensor(icl_prefix_ids[1:], dtype=torch.long)
            no_model_data["loss_mask"][i][:input_len] = 1.0
            no_model_data["input_lens"][i] = input_len
            no_model_data["idxs"][i] = samp["sid"]
            
            last_token = icl_prefix_ids[-1]
            s = max_length
            no_model_data["pos_mask"][i][s-1] = True
            for yn_id in self.all_yn[self.data_name]:
                l = len(yn_id)
                model_data["input_ids"][i][s:s+l] = torch.tensor([last_token] + yn_id[:-1], dtype=torch.long)
                model_data["attention_mask"][i][s:s+l, :input_len] = 1.0
                model_data["attention_mask"][i][s:s+l, s:s+l] = torch.tril(torch.ones(l, l))
                model_data["position_ids"][i][s:s+l] = torch.arange(input_len, input_len+l)
                no_model_data["label"][i][s:s+l] = torch.tensor(yn_id, dtype=torch.long)
                no_model_data["loss_mask"][i][s:s+l] = 1.0
                s += l
                no_model_data["pos_mask"][i][s-1] = True
            no_model_data["yn_label"][i] = self.all_yn[self.data_name].index(samp["target_ids"])
        
        return model_data, no_model_data

    def move_to_device(self, model_data, no_model_data, device):
        for k in model_data:
            if model_data[k] is not None:
                model_data[k] = model_data[k].to(device)

        for k in no_model_data:
            if isinstance(no_model_data[k], torch.Tensor):
                no_model_data[k] = no_model_data[k].to(device)



def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_config)
    
    return tokenizer


def get_model(args, device):
    model = GPTNeoForCausalLM.from_pretrained(args.model_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
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


def evaluate_gen(args, tokenizer: GPT2Tokenizer, model: GPTNeoForCausalLM, dataset: NIEvalDataset, epoch, device):
    
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
                        
            outputs = model(**model_batch, output_attentions=False)
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


def evaluate_yn(args, tokenizer: GPT2Tokenizer, model: GPTNeoForCausalLM, dataset: NIEvalDataset, epoch, device):
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
            outputs = model(**model_batch, output_attentions=args.output_attentions)
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


def evaluate_all(args, tokenizer, model, dataset: NIEvalDataset, split, epoch, device):
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

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = args.gradient_accumulation_steps
    
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    
    dataset = NIEvalDataset(args, tokenizer, args.data_dir)
    
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
        
    if args.do_eval:
        evaluate_all(args, tokenizer, model, dataset, "test", 0, device)
        compute_score(args.ni_ref_file, args.save)


if __name__ == "__main__":
    print(os.environ["CODE_BASE"])
    main()
