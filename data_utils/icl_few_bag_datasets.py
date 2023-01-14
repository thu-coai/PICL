import random
import torch
    
from .icl_datasets import ICLTrainDataset, ICLEvalDataset


class FewBagICLTrainDataset(ICLTrainDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=True, build_train_idx=True):
        super().__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool, build_train_idx)

    def _get_max_demo_length_rand(self, samp):
        if self.max_length_all_demos is not None:
            return self.max_length_all_demos
        else:
            assert self.max_length is not None
            test_samp_length = len(samp["context_ids"]) + len(samp["target_ids"])
            return self.max_length - test_samp_length

    def collate(self, samples):
        bs = len(samples)

        if self.max_length_all_demos is not None:
            ml = self.max_length_all_demos + self.max_length_per_sample
        else:
            ml = self.max_length

        model_data = {
            "input_ids": torch.ones(bs, ml, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, ml, ml),
            "position_ids": torch.zeros(bs, ml, dtype=torch.long),
        }
        no_model_data = {
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "label": torch.ones(bs, ml, dtype=torch.long) * -100,
            "loss_mask": torch.zeros(bs, ml)
        }

        for i, samp in enumerate(samples):
            icl_prefix_ids = [x for y in samp["all_demo_ids"] for x in y] + samp["context_ids"]
            input_len = len(icl_prefix_ids) + len(samp["target_ids"]) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(icl_prefix_ids + samp["target_ids"][:-1], dtype=torch.long)
            st = 0
            max_demo_len = max(map(len, samp["all_demo_ids"]), default=0)
            for demo_ids in samp["all_demo_ids"]:
                model_data["attention_mask"][i][st:st+len(demo_ids),st:st+len(demo_ids)] = torch.tril(torch.ones(len(demo_ids), len(demo_ids)))
                if self.args.pos_type == 0:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(0, len(demo_ids), dtype=torch.long)
                elif self.args.pos_type == 1:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                elif self.args.pos_type == 2:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                elif self.args.pos_type == 3:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                    model_data["position_ids"][i][st] = 0
                st += len(demo_ids)
            model_data["attention_mask"][i][st:input_len, 0:st] = 1
            model_data["attention_mask"][i][st:input_len, st:input_len] = torch.tril(torch.ones(input_len - st, input_len - st))
            if self.args.pos_type == 0:
                model_data["position_ids"][i][st:input_len] = torch.arange(0, input_len-st, dtype=torch.long)
            elif self.args.pos_type in [1,2,3]:
                model_data["position_ids"][i][st:input_len] = torch.arange(0, input_len-st, dtype=torch.long) + max_demo_len

            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            if self.args.icl_sup == "test_full":
                no_model_data["label"][i][len(icl_prefix_ids)-len(samp["context_ids"]):input_len] = torch.tensor(samp["context_ids"][1:] + samp["target_ids"], dtype=torch.long)
            elif self.args.icl_sup == "test_target":
                no_model_data["label"][i][len(icl_prefix_ids)-1:input_len] = torch.tensor(samp["target_ids"], dtype=torch.long)
            else:
                raise NotImplementedError
            no_model_data["loss_mask"][i][len(icl_prefix_ids)-1:input_len] = 1.0
        
        return model_data, no_model_data


class FewBagICLEvalDataset(ICLEvalDataset):
    def __init__(self, args, tokenizer, path, split, num, ratio, shot, rng_sample: random.Random, rng_order: random.Random, as_pool=False):
        super().__init__(args, tokenizer, path, split, num, ratio, shot, rng_sample, rng_order, as_pool)

    def _get_max_demo_length_rand(self, samp):
        if self.max_length_all_demos is not None:
            return self.max_length_all_demos
        else:
            assert self.max_length is not None
            if len(samp["options_ids"]) > 0:
                test_samp_length = len(samp["context_ids"]) + max([len(x) for x in samp["options_ids"]])
            else:
                test_samp_length = len(samp["context_ids"]) + len(samp["target_ids"])
            
            return self.max_length - test_samp_length

    def _get_max_demo_length_share(self, pool_d):
        if self.max_length_all_demos is not None:
            return self.max_length_all_demos
        else:
            assert self.max_length is not None
            return self.max_length - max([len(dd["context_ids"]) + max(map(len, dd["options_ids"])) for dd in pool_d])
        
    def collate(self, samples):
        bs = len(samples)
        max_options_sizes = max(self.option_sizes[self.data_name][self.prompt_name])
        
        if self.max_length_all_demos is not None:
            ml = self.valid_max_length_all_demos[self.data_name][self.prompt_name] + self.valid_max_length_per_sample[self.data_name][self.prompt_name]
        else:
            ml = self.valid_max_length[self.data_name][self.prompt_name]
        
        # ml has contained max_options_sizes, but is large enough
                
        model_data = {
            "input_ids": torch.ones(bs, ml + max_options_sizes, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, ml + max_options_sizes, ml + max_options_sizes),
            "position_ids": torch.zeros(bs, ml + max_options_sizes, dtype=torch.long),
        }
        no_model_data = {
            "idxs": torch.zeros(bs, 3, dtype=torch.long),
            "option_label": torch.ones(bs, dtype=torch.long),
            "label": torch.ones(bs, ml + max_options_sizes, dtype=torch.long) * (-100),
            "loss_mask": torch.zeros(bs, ml + max_options_sizes),
            "pos_mask": torch.zeros(bs, ml + max_options_sizes, dtype=torch.bool),
            "icl_sample_lens": torch.ones(bs, self.shot + 1, dtype=torch.long) * -1
        }

        for i, samp in enumerate(samples):
            icl_prefix_ids = [x for y in samp["all_demo_ids"] for x in y] + samp["context_ids"]
            input_len = len(icl_prefix_ids) - 1
            model_data["input_ids"][i][:input_len] = torch.tensor(icl_prefix_ids[:-1], dtype=torch.long) # we move the last token of the prefix to the begining of each option
            st = 0
            max_demo_len = max(map(len, samp["all_demo_ids"]), default=0)
            for did, demo_ids in enumerate(samp["all_demo_ids"]):
                model_data["attention_mask"][i][st:st+len(demo_ids),st:st+len(demo_ids)] = torch.tril(torch.ones(len(demo_ids), len(demo_ids)))
                if self.args.pos_type == 0:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(0, len(demo_ids), dtype=torch.long)
                elif self.args.pos_type == 1:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                elif self.args.pos_type == 2:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                elif self.args.pos_type == 3:
                    model_data["position_ids"][i][st:st+len(demo_ids)] = torch.arange(max_demo_len - len(demo_ids), max_demo_len, dtype=torch.long)
                    model_data["position_ids"][i][st] = 0
                st += len(demo_ids)
                no_model_data["icl_sample_lens"][i][did] = len(demo_ids)

            model_data["attention_mask"][i][st:input_len, 0:st] = 1
            model_data["attention_mask"][i][st:input_len, st:input_len] = torch.tril(torch.ones(input_len-st, input_len-st))
            if self.args.pos_type == 0:
                model_data["position_ids"][i][st:input_len] = torch.arange(0, input_len-st, dtype=torch.long)
            elif self.args.pos_type in [1,2,3]:
                model_data["position_ids"][i][st:input_len] = torch.arange(0, input_len-st, dtype=torch.long) + max_demo_len
            
            no_model_data["idxs"][i] = torch.tensor(samp["idxs"], dtype=torch.long)
            last_token = icl_prefix_ids[-1]
            st2 = ml
            no_model_data["pos_mask"][i][st2-1] = True
            for option_id in samp["options_ids"]:
                l = len(option_id)
                model_data["input_ids"][i][st2:st2+l] = torch.tensor([last_token] + option_id[:-1], dtype=torch.long)
                model_data["attention_mask"][i][st2:st2+l, :input_len] = 1.0
                model_data["attention_mask"][i][st2:st2+l, st2:st2+l] = torch.tril(torch.ones(l, l))
                if self.args.pos_type == 0:
                    model_data["position_ids"][i][st2:st2+l] = torch.arange(input_len-st, input_len-st+l)
                elif self.args.pos_type in [1,2,3]:
                    model_data["position_ids"][i][st2:st2+l] = torch.arange(input_len-st, input_len-st+l) + max_demo_len

                no_model_data["label"][i][st2:st2+l] = torch.tensor(option_id, dtype=torch.long)
                no_model_data["loss_mask"][i][st2:st2+l] = 1.0
                st2 += l
                no_model_data["pos_mask"][i][st2-1] = True
            
            assert max(model_data["position_ids"][i]) < 1024, max(model_data["position_ids"][i])
            
            no_model_data["option_label"][i] = samp["option_label_id"]
        
        return model_data, no_model_data

