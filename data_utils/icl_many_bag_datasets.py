import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader

from .icl_datasets import ICLTrainDataset, ICLEvalDataset


class ICLPtInnerTrainDataset(Dataset):
    def __init__(self, args, chunk_len, pad_id, icl_batch):
        super(ICLPtInnerTrainDataset, self).__init__()
        self.args = args
        self.pad_id = pad_id
        self.data = icl_batch["input_ids"]
        self.lengths = icl_batch["lengths"]
        self.valid_lengths = icl_batch["valid_lengths"]
        self.pos = icl_batch["pos"]
        self.demo_start_pos = icl_batch["demo_start_pos"]
        self.demo_pos_shift = icl_batch["demo_pos_shift"]
        self.max_length = chunk_len
        assert len(self.data) == len(self.lengths)
        assert len(self.data) == len(self.pos)
        assert len(self.data) == len(self.demo_pos_shift)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx], self.pos[idx], self.demo_pos_shift[idx], self.demo_start_pos[idx], self.valid_lengths[idx]
    
    def collate(self, samples):
        bs = len(samples)
        model_batch = {
            "input_ids": torch.ones(bs, self.max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length, self.max_length),
            "position_ids": torch.zeros(bs, self.max_length, dtype=torch.long),
        }
        
        no_model_batch = {
            "lengths": torch.zeros(bs, dtype=torch.long),
            "valid_lengths": torch.zeros(bs, dtype=torch.long),
            "pos": torch.zeros(bs, 2, dtype=torch.long),
            "demo_start_pos": torch.zeros(bs, 1, dtype=torch.long)
        }
        
        for i, (s,l,p,ps,dsp,vl) in enumerate(samples):
            assert l <= ps
            model_batch["input_ids"][i][:l] = torch.tensor(s, dtype=torch.long)
            model_batch["attention_mask"][i][:l, :l] = torch.tril(torch.ones(l, l))
            if self.args.pos_type in [0,1]:
                model_batch["position_ids"][i][:l] = torch.arange(0, l, dtype=torch.long)
            elif self.args.pos_type == 2:
                model_batch["position_ids"][i][:l] = torch.arange(ps-l, ps, dtype=torch.long)
            elif self.args.pos_type == 3:
                model_batch["position_ids"][i][:l] = torch.arange(ps-l, ps, dtype=torch.long)
                model_batch["position_ids"][i][0] = 0

            no_model_batch["lengths"][i] = l
            no_model_batch["valid_lengths"][i] = vl
            no_model_batch["pos"][i] = torch.tensor(p, dtype=torch.long)
            no_model_batch["demo_start_pos"][i] = dsp
            
        return model_batch, no_model_batch


def _collate_demo_out_model(args, all_demo_ids_full_batch, bs, max_length_per_sample, max_length_all_demos, chunk_len, pad_id):
    demo_batch = {
        "input_ids": [],
        "lengths": [],
        "valid_lengths": [],
        "pos": [],
        "demo_start_pos": [],
        "max_length_all_demos": max_length_all_demos,
        "demo_pos_shift": [],
        "test_pos_shift": []
    }
    for i, all_demo_ids in enumerate(all_demo_ids_full_batch):
        demo_batch["input_ids"].extend(all_demo_ids)
        demo_batch["lengths"].extend([len(x) for x in all_demo_ids])
        if args.add_bos and args.remove_inner_bos:
            demo_batch["valid_lengths"].extend([len(x)-1 for x in all_demo_ids])
            demo_batch["demo_start_pos"].extend([1 for _ in all_demo_ids])
        else:
            demo_batch["valid_lengths"].extend([len(x) for x in all_demo_ids])
            demo_batch["demo_start_pos"].extend([0 for _ in all_demo_ids])
        st = 0
        max_demo_pos = max(map(len, all_demo_ids), default=0)
        # print(max_demo_pos)
        
        # naive many
        for ids in all_demo_ids:
            demo_batch["pos"].append((i, st))
            if args.add_bos and args.remove_inner_bos:
                st += len(ids)-1
            else:
                st += len(ids)
            demo_batch["demo_pos_shift"].append(max_demo_pos)
        
        demo_batch["test_pos_shift"].append(max_demo_pos)
        
        assert max_demo_pos + max_length_per_sample <= args.gpt_max_length, (max_demo_pos, max_length_per_sample) # max position ids

    inner_dataset = ICLPtInnerTrainDataset(args, chunk_len, pad_id, demo_batch)
    inner_dataloader = DataLoader(inner_dataset, batch_size=args.icl_inner_batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=inner_dataset.collate)
    inner_inputs = [(mb, nmb) for mb, nmb in inner_dataloader]

    return demo_batch, inner_inputs


def _collate_demo_in_model(args, all_demo_ids_full_batch, bs, max_length_per_sample, max_length_all_demos, chunk_len, pad_id):
    num_all_demos = sum(map(len, all_demo_ids_full_batch))
    demo_batch = {
        "select_index": torch.zeros(bs, max_length_all_demos, dtype=torch.long),
        "context_select_index": torch.zeros(bs, max_length_all_demos, dtype=torch.long),
        "select_index_inv": [],
        "max_length_all_demos": max_length_all_demos,
        "demo_pos_shift": [],
        "test_pos_shift": [],
        "compress_input_ids": torch.ones(bs, max_length_all_demos, dtype=torch.long) * pad_id,
        "compress_position_ids": torch.zeros(bs, max_length_all_demos, dtype=torch.long),
        "attention_mask": torch.zeros(num_all_demos, chunk_len, chunk_len),
    }
    n_demo = 0
    for i, all_demo_ids in enumerate(all_demo_ids_full_batch):
        st = 0
        max_demo_pos = max(map(len, all_demo_ids), default=0)
        # print(max_demo_pos)

        # in model many
        _select_index = []
        _context_select_index = []
        for ids in all_demo_ids:
            demo_batch["compress_input_ids"][i][st:st+len(ids)] = torch.tensor(ids, dtype=torch.long)
            if args.pos_type in [0,1]:
                demo_batch["compress_position_ids"][i][st:st+len(ids)] = torch.arange(0, len(ids), dtype=torch.long)
            elif args.pos_type == 2:
                demo_batch["compress_position_ids"][i][st:st+len(ids)] = torch.arange(max_demo_pos-len(ids), max_demo_pos, dtype=torch.long)
            elif args.pos_type == 3:
                demo_batch["compress_position_ids"][i][st:st+len(ids)] = torch.arange(max_demo_pos-len(ids), max_demo_pos, dtype=torch.long)
                demo_batch["compress_position_ids"][i][st] = 0
            else:
                raise NotImplementedError
            
            demo_batch["attention_mask"][n_demo][:len(ids),:len(ids)] = torch.tril(torch.ones(len(ids), len(ids)))
            
            _select_index.extend([n_demo*chunk_len + idx for idx in range(len(ids))])
            if args.add_bos and args.remove_inner_bos:
                _context_select_index.extend([n_demo*chunk_len + idx for idx in range(1,len(ids))])
            else:
                _context_select_index.extend([n_demo*chunk_len + idx for idx in range(0,len(ids))])
            demo_batch["select_index_inv"].append([i*max_length_all_demos+st+idx for idx in range(len(ids))] + [0] * (chunk_len - len(ids)))
            
            # other
            st += len(ids)
            demo_batch["demo_pos_shift"].append(max_demo_pos)
            n_demo += 1
        
        demo_batch["test_pos_shift"].append(max_demo_pos)
        demo_batch["select_index"][i][:len(_select_index)] = torch.tensor(_select_index, dtype=torch.long)
        demo_batch["context_select_index"][i][:len(_context_select_index)] = torch.tensor(_context_select_index, dtype=torch.long)
        
        assert max_demo_pos + max_length_per_sample < args.gpt_max_length, (max_demo_pos, max_length_per_sample) # max position ids

    demo_batch["select_index_inv"] = torch.tensor(demo_batch["select_index_inv"], dtype=torch.long)

    inner_inputs = {
        "demo_input_ids": demo_batch["compress_input_ids"],
        "demo_index": demo_batch["select_index"],
        "demo_context_index": demo_batch["context_select_index"],
        "demo_index_inv": demo_batch["select_index_inv"],
        "demo_attention_mask": demo_batch["attention_mask"],
        "demo_position_ids": demo_batch["compress_position_ids"],
    }

    return demo_batch, inner_inputs


class ManyBagICLTrainDataset(ICLTrainDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=True, build_train_idx=True):
        super().__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool, build_train_idx)

    def _get_max_demo_length_rand(self, samp):
        return self.max_length_all_demos

    def collate(self, samples):
        bs = len(samples)
        model_batch = {
            "input_ids": torch.ones(bs, self.max_length_per_sample, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length_per_sample, self.max_length_per_sample + self.max_length_all_demos, dtype=torch.long),
            "position_ids": torch.zeros(bs, self.max_length_per_sample, dtype=torch.long),
            "attn_scale_factor": torch.ones(bs, 1, self.max_length_per_sample, self.max_length_per_sample + self.max_length_all_demos) if self.args.attn_scale else None
        }
        no_model_batch = {
            "label": torch.ones(bs, self.max_length_per_sample, dtype=torch.long) * (-100),
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "loss_mask": torch.zeros(bs, self.max_length_per_sample)
        }
        chunk_len = self.chunk_len if self.chunk_len > 0 else self.max_length_per_sample
        
        _collate_demo = _collate_demo_in_model if self.args.icl_many_in_model else _collate_demo_out_model
        demo_batch, inner_inputs = _collate_demo(self.args, [samp["all_demo_ids"] for samp in samples], bs, self.max_length_per_sample, self.max_length_all_demos, chunk_len, self.pad_id)
        
        assert len(demo_batch["test_pos_shift"]) == len(samples)
        for i, samp in enumerate(samples):
            input_len = len(samp["context_ids"]) + len(samp["target_ids"]) - 1
            if self.args.add_bos and self.args.remove_inner_bos:
                demos_len = sum([len(x)-1 for x in samp["all_demo_ids"]])
            else:
                demos_len = sum([len(x) for x in samp["all_demo_ids"]])
            model_batch["input_ids"][i][:input_len] = torch.tensor(samp["context_ids"] + samp["target_ids"][:-1], dtype=torch.long)

            if self.args.add_bos and self.args.remove_inner_bos:
                model_batch["attention_mask"][i][0, 0] = 1.0
                model_batch["attention_mask"][i][1:input_len, :demos_len+1] = 1.0
                attn_mask = torch.tril(torch.ones(input_len-1, input_len-1))
                model_batch["attention_mask"][i][1:input_len, self.max_length_all_demos+1:self.max_length_all_demos+input_len] = attn_mask
                if self.args.attn_scale:
                    model_batch["attn_scale_factor"][i][0][1:input_len, self.max_length_all_demos+1:self.max_length_all_demos+input_len] = attn_mask * len(samp["all_demo_ids"]) + (1 - attn_mask)
            else:
                model_batch["attention_mask"][i][:input_len, :demos_len] = 1.0
                attn_mask = torch.tril(torch.ones(input_len, input_len))
                model_batch["attention_mask"][i][:input_len, self.max_length_all_demos:self.max_length_all_demos+input_len] = attn_mask
                if self.args.attn_scale:
                    model_batch["attn_scale_factor"][i][0][:input_len, self.max_length_all_demos:self.max_length_all_demos+input_len] = attn_mask * len(samp["all_demo_ids"]) + (1 - attn_mask)

            if self.args.pos_type == 0:
                model_batch["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            elif self.args.pos_type in [1,2]:
                model_batch["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
                model_batch["position_ids"][i][:input_len] += demo_batch["test_pos_shift"][i]
            elif self.args.pos_type == 3:
                model_batch["position_ids"][i][1:input_len] = torch.arange(0, input_len-1, dtype=torch.long)
                model_batch["position_ids"][i][1:input_len] += demo_batch["test_pos_shift"][i]
                model_batch["position_ids"][i][0] = 0
            else:
                raise NotImplementedError()
            no_model_batch["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            if self.args.icl_sup == "test_full":
                no_model_batch["label"][i][:input_len] = torch.tensor(samp["context_ids"][1:] + samp["target_ids"], dtype=torch.long)
            elif self.args.icl_sup == "test_target":
                no_model_batch["label"][i][len(samp["context_ids"])-1:input_len] = torch.tensor(samp["target_ids"], dtype=torch.long)
            else:
                raise NotImplementedError
            no_model_batch["loss_mask"][i][len(samp["context_ids"])-1:input_len] = 1.0
        
        return model_batch, no_model_batch, demo_batch, inner_inputs


class ManyBagICLEvalDataset(ICLEvalDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=False):
        super().__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool)

    def _get_max_demo_length_rand(self, samp):
        return self.max_length_all_demos

    def _get_max_demo_length_share(self, pool_d):
        return self.max_length_all_demos

    def get_demo_batch(self):
        # NOTE: for shared, icl batch is gotten from this function, rather than collate
        all_demo_ids = self.__getitem__(0)["all_demo_ids"]
        max_length_per_sample = self.valid_max_length_per_sample[self.data_name][self.prompt_name]
        max_length_all_demos = self.valid_max_length_all_demos[self.data_name][self.prompt_name]
        chunk_len = self.chunk_len if self.chunk_len > 0 else self.max_length_per_sample # for demo, so is self.max_length_per_sample (not max_length_per_sample)
        _collate_demo = _collate_demo_in_model if self.args.icl_many_in_model else _collate_demo_out_model
        demo_batch, inner_inputs = _collate_demo(self.args, [all_demo_ids], 1, max_length_per_sample, max_length_all_demos, chunk_len, self.pad_id)

        return demo_batch, inner_inputs
    
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def collate(self, samples):
        bs = len(samples)
        max_options_sizes = max(self.option_sizes[self.data_name][self.prompt_name])
        max_length_per_sample = self.valid_max_length_per_sample[self.data_name][self.prompt_name]
        max_length_all_demos = self.valid_max_length_all_demos[self.data_name][self.prompt_name]

        # max_length_per_sample has contained max_options_sizes, but is large enough
        # NOTE: max_length_per_sample is only applicable to test samples, not demos!!! (demos should use self.max_length_per_sample)

        model_data = {
            "input_ids": torch.ones(bs, max_length_per_sample + max_options_sizes, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length_per_sample + max_options_sizes, max_length_all_demos + max_length_per_sample + max_options_sizes),
            "position_ids": torch.zeros(bs, max_length_per_sample + max_options_sizes, dtype=torch.long),
            "attn_scale_factor": torch.ones(bs, 1, max_length_per_sample + max_options_sizes, max_length_all_demos + max_length_per_sample + max_options_sizes) if self.args.attn_scale else None,
        }
        no_model_data = {
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "option_label": torch.ones(bs, dtype=torch.long),
            "label": torch.ones(bs, max_length_per_sample + max_options_sizes, dtype=torch.long) * (-100),
            "loss_mask": torch.zeros(bs, max_length_per_sample + max_options_sizes),
            "pos_mask": torch.zeros(bs, max_length_per_sample + max_options_sizes, dtype=torch.bool),
            "all_demo_ids": [samp["all_demo_ids"] for samp in samples]
        }

        chunk_len = self.chunk_len if self.chunk_len > 0 else self.max_length_per_sample # for demo, so is self.max_length_per_sample (not max_length_per_sample)
        
        _collate_demo = _collate_demo_in_model if self.args.icl_many_in_model else _collate_demo_out_model
        demo_batch, inner_inputs = _collate_demo(self.args, [samp["all_demo_ids"] for samp in samples], bs, max_length_per_sample, max_length_all_demos, chunk_len, self.pad_id)
        
        assert len(demo_batch["test_pos_shift"]) == len(samples)
        for i, samp in enumerate(samples):
            input_len = len(samp["context_ids"]) - 1
            if self.args.add_bos and self.args.remove_inner_bos:
                demos_len = sum([len(x)-1 for x in samp["all_demo_ids"]])
            else:
                demos_len = sum([len(x) for x in samp["all_demo_ids"]])
            model_data["input_ids"][i][:input_len] = torch.tensor(samp["context_ids"][:-1], dtype=torch.long) # we move the last token of the prefix to the begining of each option
            
            if self.args.add_bos and self.args.remove_inner_bos:
                model_data["attention_mask"][i][0, 0] = 1.0
                model_data["attention_mask"][i][1:input_len, :demos_len+1] = 1.0
                attn_mask = torch.tril(torch.ones(input_len-1, input_len-1))
                model_data["attention_mask"][i][1:input_len, max_length_all_demos+1:max_length_all_demos+input_len] = attn_mask
                if self.args.attn_scale:
                    model_data["attn_scale_factor"][i][0][1:input_len, max_length_all_demos+1:max_length_all_demos+input_len] = attn_mask * len(samp["all_demo_ids"]) + (1 - attn_mask)
            else:
                model_data["attention_mask"][i][:input_len, :demos_len] = 1.0
                attn_mask = torch.tril(torch.ones(input_len, input_len))
                model_data["attention_mask"][i][:input_len, max_length_all_demos:max_length_all_demos+input_len] = attn_mask
                if self.args.attn_scale:
                    model_data["attn_scale_factor"][i][0][:input_len, max_length_all_demos:max_length_all_demos+input_len] = attn_mask * len(samp["all_demo_ids"]) + (1 - attn_mask)

            if self.args.pos_type == 0:
                model_data["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            elif self.args.pos_type in [1,2]:
                model_data["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
                model_data["position_ids"][i][:input_len] += demo_batch["test_pos_shift"][i]
            elif self.args.pos_type == 3:
                model_data["position_ids"][i][1:input_len] = torch.arange(0, input_len-1, dtype=torch.long)
                model_data["position_ids"][i][1:input_len] += demo_batch["test_pos_shift"][i]
                model_data["position_ids"][i][0] = 0
            else:
                raise NotImplementedError()
            
            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            
            # options
            last_token = samp["context_ids"][-1]
            st = max_length_per_sample
            no_model_data["pos_mask"][i][st-1] = True
            for option_id in samp["options_ids"]:
                l = len(option_id)
                model_data["input_ids"][i][st:st+l] = torch.tensor([last_token] + option_id[:-1], dtype=torch.long)
                
                if self.args.add_bos and self.args.remove_inner_bos:
                    model_data["attention_mask"][i][st:st+l, :demos_len+1] = 1.0
                    model_data["attention_mask"][i][st:st+l, max_length_all_demos+1:max_length_all_demos+input_len] = 1.0
                    attn_mask = torch.tril(torch.ones(l, l))
                    model_data["attention_mask"][i][st:st+l, max_length_all_demos+st:max_length_all_demos+st+l] = attn_mask
                    if self.args.attn_scale:
                        model_data["attn_scale_factor"][i][0][st:st+l, max_length_all_demos+1:max_length_all_demos+input_len] = len(samp["all_demo_ids"])
                        model_data["attn_scale_factor"][i][0][st:st+l, max_length_all_demos+st:max_length_all_demos+st+l] = attn_mask * len(samp["all_demo_ids"]) + (1 - attn_mask)
                else:
                    model_data["attention_mask"][i][st:st+l, :demos_len] = 1.0
                    model_data["attention_mask"][i][st:st+l, max_length_all_demos:max_length_all_demos+input_len] = 1.0
                    attn_mask = torch.tril(torch.ones(l, l))
                    model_data["attention_mask"][i][st:st+l, max_length_all_demos+st:max_length_all_demos+st+l] = attn_mask
                    if self.args.attn_scale:
                        model_data["attn_scale_factor"][i][0][st:st+l, max_length_all_demos+st:max_length_all_demos+st+l] = attn_mask * len(samp["all_demo_ids"]) + (1 - attn_mask)
                        model_data["attn_scale_factor"][i][0][st:st+l, max_length_all_demos:max_length_all_demos+input_len] = len(samp["all_demo_ids"])

                if self.args.pos_type == 0:
                    model_data["position_ids"][i][st:st+l] = torch.arange(input_len, input_len+l)
                elif self.args.pos_type in [1,2]:
                    model_data["position_ids"][i][st:st+l] = torch.arange(input_len, input_len+l)
                    model_data["position_ids"][i][st:st+l] += demo_batch["test_pos_shift"][i]
                elif self.args.pos_type == 3:
                    model_data["position_ids"][i][st:st+l] = torch.arange(input_len-1, input_len-1+l)
                    model_data["position_ids"][i][st:st+l] += demo_batch["test_pos_shift"][i]
                else:
                    raise NotImplementedError

                no_model_data["label"][i][st:st+l] = torch.tensor(option_id, dtype=torch.long)
                no_model_data["loss_mask"][i][st:st+l] = 1.0
                st += l
                no_model_data["pos_mask"][i][st-1] = True
            no_model_data["option_label"][i] = samp["option_label_id"]

        return model_data, no_model_data, demo_batch, inner_inputs
