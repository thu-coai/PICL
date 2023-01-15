import random
import torch
import os
import re
import json
import pickle
from torch.utils.data import Dataset

from .icl_datasets import ICLEvalDataset
from .sni_tasks import NI_TASKS, YN_TASKS
from utils import get_rank, print_rank, barrier


class ICLEvalCLSDataset(ICLEvalDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=False):
        super().__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool)

    def _get_max_demo_length_rand(self, samp):
        if len(samp["options_ids"]) > 0:
            test_samp_length = len(samp["context_ids"]) + max([len(x) for x in samp["options_ids"]])
        else:
            test_samp_length = len(samp["context_ids"]) + len(samp["target_ids"])
        
        return self.max_length - test_samp_length

    def _get_max_demo_length_share(self, pool_d):
        return self.max_length - (max([len(dd["context_ids"]) + (max(map(len, dd["options_ids"])) if dd["options_ids"] is not None else len(dd["target_ids"])) for dd in pool_d]))

    def collate(self, samples):
        bs = len(samples)
        max_options_sizes = max(self.option_sizes[self.data_name][self.prompt_name])
        max_length = self.valid_max_length[self.data_name][self.prompt_name]
        
        # max_length has contained max_options_sizes, but is large enough
        
        model_data = {
            "input_ids": torch.ones(bs, max_length + max_options_sizes, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length + max_options_sizes, max_length + max_options_sizes),
            "position_ids": torch.zeros(bs, max_length + max_options_sizes, dtype=torch.long),
        }
        no_model_data = {
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "option_label": torch.ones(bs, dtype=torch.long),
            "label": torch.ones(bs, max_length + max_options_sizes, dtype=torch.long) * (-100),
            "loss_mask": torch.zeros(bs, max_length + max_options_sizes),
            "pos_mask": torch.zeros(bs, max_length + max_options_sizes, dtype=torch.bool),
            "icl_demo_lens": torch.ones(bs, self.shot + 1, dtype=torch.long) * -1,
            "all_demo_ids": [samp["all_demo_ids"] for samp in samples],
            "input_lens": torch.zeros(bs, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            if self.args.add_bos:
                icl_prefix_ids = [self.tokenizer.bos_token_id] + [x for y in samp["all_demo_ids"] for x in y[1:]] + samp["context_ids"][1:]
            else:
                icl_prefix_ids = [x for y in samp["all_demo_ids"] for x in y] + samp["context_ids"]
                
            input_len = len(icl_prefix_ids) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(icl_prefix_ids[:-1], dtype=torch.long) # we move the last token of the prefix to the begining of each option
            model_data["attention_mask"][i][:input_len, :input_len] = torch.tril(torch.ones(input_len, input_len))
            model_data["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            no_model_data["label"][i][:input_len] = torch.tensor(icl_prefix_ids[1:], dtype=torch.long)
            no_model_data["loss_mask"][i][:input_len] = 1.0
            no_model_data["input_lens"][i] = input_len
            
            for did, demo_ids in enumerate(samp["all_demo_ids"]):
                no_model_data["icl_demo_lens"][i][did] = len(demo_ids)

            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            last_token = icl_prefix_ids[-1]
            s = max_length
            no_model_data["pos_mask"][i][s-1] = True
            for option_id in samp["options_ids"]:
                l = len(option_id)
                model_data["input_ids"][i][s:s+l] = torch.tensor([last_token] + option_id[:-1], dtype=torch.long)
                model_data["attention_mask"][i][s:s+l, :input_len] = 1.0
                model_data["attention_mask"][i][s:s+l, s:s+l] = torch.tril(torch.ones(l, l))
                model_data["position_ids"][i][s:s+l] = torch.arange(input_len, input_len+l)
                no_model_data["label"][i][s:s+l] = torch.tensor(option_id, dtype=torch.long)
                no_model_data["loss_mask"][i][s:s+l] = 1.0
                s += l
                no_model_data["pos_mask"][i][s-1] = True
            no_model_data["option_label"][i] = samp["option_label_id"]
        
        return model_data, no_model_data


class ICLEvalSNIDataset(Dataset):
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
        return self.data_name in YN_TASKS

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
            if dn in YN_TASKS:
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
