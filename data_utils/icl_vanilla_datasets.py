import random
import torch
    
from .icl_datasets import ICLTrainDataset, ICLEvalDataset


class VanillaICLTrainDataset(ICLTrainDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=True, build_train_idx=True):
        super().__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool, build_train_idx)
    
    def _get_max_demo_length_rand(self, samp):
        test_samp_length = len(samp["context_ids"]) + len(samp["target_ids"])
        return self.max_length - test_samp_length
    
    def collate(self, samples):
        bs = len(samples)

        model_data = {
            "input_ids": torch.ones(bs, self.max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_length, self.max_length),
            "position_ids": torch.zeros(bs, self.max_length, dtype=torch.long),
        }
        no_model_data = {
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "label": torch.ones(bs, self.max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, self.max_length)
        }

        for i, samp in enumerate(samples):
            icl_prefix_ids = [x for y in samp["all_demo_ids"] for x in y] + samp["context_ids"]
            input_len = len(icl_prefix_ids) + len(samp["target_ids"]) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(icl_prefix_ids + samp["target_ids"][:-1], dtype=torch.long)
            model_data["attention_mask"][i][:input_len, :input_len] = torch.tril(torch.ones(input_len, input_len))
            model_data["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            if self.args.icl_sup == "all_full":
                no_model_data["label"][i][:input_len] = torch.tensor(icl_prefix_ids[1:] + samp["target_ids"], dtype=torch.long)
                no_model_data["loss_mask"][i][:input_len] = 1.0
            elif self.args.icl_sup == "test_full":
                no_model_data["label"][i][len(icl_prefix_ids)-len(samp["context_ids"]):input_len] = torch.tensor(samp["context_ids"][1:] + samp["target_ids"], dtype=torch.long)
                no_model_data["loss_mask"][i][len(icl_prefix_ids)-len(samp["context_ids"]):input_len] = 1.0
            elif self.args.icl_sup == "all_target":
                icl_prefix_ids_only_target = [x for y in samp["all_demo_ids_only_target"] for x in y] + [-100] * len(samp["context_ids"])
                no_model_data["label"][i][:input_len] = torch.tensor(icl_prefix_ids_only_target[1:] + samp["target_ids"], dtype=torch.long)
                # no_model_data["loss_mask"][i][len(icl_prefix_ids)-1:input_len] = 1.0
                assert 1 == 0, "all target loss mask not complete"
            elif self.args.icl_sup == "test_target":
                no_model_data["label"][i][len(icl_prefix_ids)-1:input_len] = torch.tensor(samp["target_ids"], dtype=torch.long)
                no_model_data["loss_mask"][i][len(icl_prefix_ids)-1:input_len] = 1.0
            else:
                raise NotImplementedError
        
        return model_data, no_model_data


class VanillaICLEvalDataset(ICLEvalDataset):
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

    def collate_gen(self, samples):
        bs = len(samples)
        max_length = self.valid_max_length[self.data_name][self.prompt_name]
        max_length_gen_context = self.valid_max_length_all_context[self.data_name][self.prompt_name]
        # max_length has contained max_options_sizes, but is large enough
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, max_length),
            "position_ids": torch.zeros(bs, max_length, dtype=torch.long),
        }
        no_model_data = {
            "gen_input_ids": torch.ones(bs, max_length_gen_context, dtype=torch.long) * self.pad_id,
            "gen_attention_mask": torch.zeros(bs, max_length_gen_context),
            "gen_position_ids": torch.zeros(bs, max_length_gen_context, dtype=torch.long),
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "label": torch.ones(bs, max_length, dtype=torch.long) * (-100),
            "loss_mask": torch.zeros(bs, max_length),
            "icl_demo_lens": torch.ones(bs, self.shot + 1, dtype=torch.long) * -1,
            "all_demo_ids": [samp["all_demo_ids"] for samp in samples],
            "input_lens": torch.zeros(bs, dtype=torch.long)
        }

        for i, samp in enumerate(samples):
            icl_prefix_ids = [x for y in samp["all_demo_ids"] for x in y] + samp["context_ids"]
            input_len = len(icl_prefix_ids) + len(samp["target_ids"]) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(icl_prefix_ids + samp["target_ids"][:-1], dtype=torch.long)
            model_data["attention_mask"][i][:input_len, :input_len] = torch.tril(torch.ones(input_len, input_len))
            model_data["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            no_model_data["label"][i][:input_len] = torch.tensor(icl_prefix_ids[1:] + samp["target_ids"], dtype=torch.long)
            no_model_data["loss_mask"][i][len(icl_prefix_ids[1:]):input_len] = 1.0
            no_model_data["input_lens"][i] = input_len
            
            for did, demo_ids in enumerate(samp["all_demo_ids"]):
                no_model_data["icl_demo_lens"][i][did] = len(demo_ids)

            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            
            no_model_data["gen_input_ids"][i][-len(icl_prefix_ids):] = torch.tensor(icl_prefix_ids, dtype=torch.long)
            no_model_data["gen_attention_mask"][i][-len(icl_prefix_ids):] = 1.0
            no_model_data["gen_position_ids"][i][-len(icl_prefix_ids):] = torch.arange(0, len(icl_prefix_ids))
        
        return model_data, no_model_data

    def collate_test(self, samples):
        bs = len(samples)
        label_size = self.label_size()
        max_length = self.real_max_length[self.data_name][self.prompt_name]
        
        model_data = {
            "input_ids": torch.ones(bs * label_size, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs * label_size, max_length, max_length),
            "position_ids": torch.zeros(bs * label_size, max_length, dtype=torch.long),
        }
        no_model_data = {
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "option_label": torch.ones(bs, dtype=torch.long),
            "label": torch.ones(bs * label_size, max_length, dtype=torch.long) * (-100),
            "loss_mask": torch.zeros(bs * label_size, max_length),
        }

        for i, samp in enumerate(samples):
            icl_prefix_ids = [x for y in samp["all_demo_ids"] for x in y] + samp["context_ids"]
            for opi, option_id in enumerate(samp["options_ids"]):
                input_len = len(icl_prefix_ids) + len(option_id) - 1
                model_data["input_ids"][label_size*i+opi][:input_len] = torch.tensor(icl_prefix_ids + option_id[:-1], dtype=torch.long) # we move the last token of the prefix to the begining of each option
                model_data["attention_mask"][label_size*i+opi][:input_len, :input_len] = torch.tril(torch.ones(input_len, input_len))
                model_data["position_ids"][label_size*i+opi][:input_len] = torch.arange(0, input_len, dtype=torch.long)                
                no_model_data["label"][label_size*i+opi][len(icl_prefix_ids)-1:input_len] = torch.tensor(option_id, dtype=torch.long)
                no_model_data["loss_mask"][label_size*i +
                                           opi][len(icl_prefix_ids)-1:input_len] = 1.0
            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            no_model_data["option_label"][i] = samp["option_label_id"]
        
        return model_data, no_model_data

    def collate_lm(self, samples):
        bs = len(samples)
        max_length = self.real_max_length[self.data_name][self.prompt_name]
        
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length, max_length),
            "position_ids": torch.zeros(bs, max_length, dtype=torch.long),
        }
        no_model_data = {
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "label": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, max_length)
        }

        for i, samp in enumerate(samples):
            icl_prefix_ids = [x for y in samp["all_demo_ids"] for x in y] + samp["context_ids"]
            input_len = len(icl_prefix_ids) + len(samp["target_ids"]) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(icl_prefix_ids + samp["target_ids"][:-1], dtype=torch.long)
            model_data["attention_mask"][i][:input_len, :input_len] = torch.tril(torch.ones(input_len, input_len))
            model_data["position_ids"][i][:input_len] = torch.arange(0, input_len, dtype=torch.long)
            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            no_model_data["label"][i][len(icl_prefix_ids)-1:input_len] = torch.tensor(samp["target_ids"], dtype=torch.long)
            no_model_data["loss_mask"][i][len(icl_prefix_ids)-1:input_len] = 1.0
        
        return model_data, no_model_data
