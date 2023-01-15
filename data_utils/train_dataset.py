import random
import torch
    
from .icl_datasets import ICLTrainDataset


class ICLTrainDataset(ICLTrainDataset):
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