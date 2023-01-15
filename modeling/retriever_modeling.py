from transformers import BertModel, RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import time
import random
import numpy as np


class RetrieverModel(nn.Module):
    def __init__(self, model_dir, vocab_size, share_model=False, pool_type="cls"):
        super(DPRModel, self).__init__()
        self.query_encoder = RobertaModel.from_pretrained(model_dir)
        self.share_model = share_model
        self.pool_type = pool_type
        if share_model:
            print('share model!')
            self.context_encoder = self.query_encoder
        else:
            self.context_encoder = RobertaModel.from_pretrained(model_dir)
            
        self.context_encoder.gradient_checkpointing_enable()

        # self.context_encoder.resize_token_embeddings(vocab_size)
        # self.query_encoder.resize_token_embeddings(vocab_size)

        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def get_repr(self, outputs, mask):
        if self.pool_type == "cls":
            return outputs[1]
        elif self.pool_type == "mean":
            return torch.sum(outputs[0] * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        else:
            raise ValueError(f"{self.pool_type} is not emplement")

    def forward(self, query_inputs, context_inputs, pos_ctx_indices=None, pos_ctx_indices_mask=None, return_loss=True):
        # encode query and contexts
        outputs = self.query_encoder(**query_inputs)
        query_repr = self.get_repr(outputs, query_inputs["attention_mask"]) # bs x d
        
        # print(f"query input ids = {query_ids['input_ids'].shape},   context input ids = {context_ids['input_ids'].shape}")
        outputs = self.context_encoder(**context_inputs)
        context_repr = self.get_repr(outputs, context_inputs["attention_mask"])

        loss = None
        sim_scores = None
        if return_loss:
            sim_scores = self.sim_score(query_repr, context_repr)
            log_softmax_scores = F.log_softmax(sim_scores, dim=-1)
            loss = -torch.sum((log_softmax_scores * pos_ctx_indices_mask), dim=-1) / torch.sum(pos_ctx_indices_mask, dim=-1)
            loss = torch.mean(loss, dim=0)
        
        return {
            "loss": loss,
            "sim_scores": sim_scores,
            "query_repr": query_repr,
            "context_repr": context_repr
        }

    def sim_score(self, query_repr, context_repr):
        scores = torch.matmul(
            query_repr, torch.transpose(context_repr, 0, 1)
        )  # num_q x num_ctx
        return scores


def get_optimizer_params(args, model: nn.Module):
    # taken from https://github.com/facebookresearch/SpanBERT/blob/0670d8b6a38f6714b85ea7a033f16bd8cc162676/code/run_tacred.py
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    return optimizer_grouped_parameters
