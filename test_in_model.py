import torch
import torch.nn as nn
import sys
import os
from transformers import GPT2Tokenizer

# print(os.environ["A"])

# icl_batch = torch.load("icl_batch.pt", map_location="cpu")
# model_batch = torch.load("model_batch.pt", map_location="cpu")

# print(icl_batch["input_ids"])
# print(icl_batch["max_icl_train_pos"])

# print(model_batch["input_ids"])
# for idx in range(len(model_batch["position_ids"])):
#     print(model_batch["position_ids"][idx].tolist())

# inner_train_model_batch = torch.load("inner_train_model_batch.pt", map_location="cpu")

# print(inner_train_model_batch["input_ids"])
# for idx in range(len(inner_train_model_batch["position_ids"])):
#     print(inner_train_model_batch["position_ids"][idx].tolist())

# few = torch.load("/home/guyuxian/CodeRepo/icl_train/few.pt", map_location="cpu")
# many = torch.load("/home/guyuxian/CodeRepo/icl_train/many.pt", map_location="cpu")

# # print(few)
# # print(many)

# x = torch.cat(list(map(torch.tensor, many[2]["input_ids"])))
# y = few[0]["input_ids"][0][:len(x)]

# print(torch.sum(torch.abs(x-y)))

# xi, x = torch.load("/home/guyuxian/CodeRepo/icl_train/hidden_states_few_unordered_3.pt", map_location="cpu")
# yi, y = torch.load("/home/guyuxian/CodeRepo/icl_train/hidden_states_many_3.pt", map_location="cpu")

# print(x.size())
# print(y.size())

# l = torch.sum(yi != 50256, dim=-1)

# y = torch.cat([yy[:ll] for yy, ll in zip(y, l)], dim=0)

# lx = torch.sum(xi != 50256)

# x = x[0][:len(y)]

# print(x.size())
# print(y.size())

# print(x)
# print(y)

# print(torch.sum(torch.abs(x - y)))

# xi = xi[0][:lx]
# yi = torch.cat([yy[:ll] for yy, ll in zip(yi, l)], dim=0)

# print(xi)
# print(yi)

# x = torch.load("/home/guyuxian/CodeRepo/icl_train/pkv_few.pt", map_location="cpu")
# y = torch.load("/home/guyuxian/CodeRepo/icl_train/pkv_many.pt", map_location="cpu")

# print(x.size())
# print(y.size())

# l = torch.sum(yi != 50256, dim=-1)

# y = torch.cat([y[:, :, bid, :, :l[bid], :] for bid in range(len(l))], dim=-2)

# lx = torch.sum(xi != 50256)

# x = x[:, :, 0, :, :y.size(-2), :]

# print(x.size())
# print(y.size())

# print(x[30][1][9][300])
# print(y[30][1][9][300])

# print(torch.sum(torch.abs(x - y)))

# l = torch.sum(yi != 50256)

# print(l)

# x = x[:, :, 0, :, :l, :]
# y = y[:, :, 0, :, :l, :]


# print(x.size())
# print(y.size())

# print(x[30][1][9][300])
# print(y[30][1][9][300])

# print(torch.sum(torch.abs(x - y)))

# lx = torch.sum(xi != 50256)

# x = torch.load("/home/guyuxian/CodeRepo/icl_train/logits_few.pt", map_location="cpu")
# y = torch.load("/home/guyuxian/CodeRepo/icl_train/logits_many.pt", map_location="cpu")

# print(x.size())
# print(y.size())

# print(x[0, lx])
# print(y[0, -1])

# x = x[0, l:lx, :]
# y = y[0]

# print(y[59])


# print(x.size())
# print(y.size())

# print(x)
# print(y)


# x = torch.load("/home/guyuxian/CodeRepo/icl_train/bt.pt", map_location="cpu")

# print(x[0]["position_ids"][0].tolist())

# mb, no_mb = torch.load("mb.pt", map_location="cpu")

# tokenizer = GPT2Tokenizer.from_pretrained("/home/guyuxian/CodeRepo/checkpoints/gpt2-large")

# print(tokenizer.decode(mb["input_ids"].cpu()[1]))

# print(mb["input_ids"][1].tolist())
# print(mb["position_ids"][1].tolist())
# # for x in mb["attention_mask"][0].int().tolist():
#     # print(x)
    
# print(no_mb["label"][1].tolist())

# print(tokenizer.encode("I love you.\n"))

# t = torch.tensor([
#     [1.0, 2, 3, 4, 0, -2],
#     [1, 2, 3, 0, -4.5, 0],
#     [1, 2, -2, -7, -10, -3],
#     [1, 2, 0, -1, 0, -8],
#     [1, 2, 3, 4, 5, 6],
# ])

# s = 3 * 7 + 1
# # index = torch.tensor([0, 1, 2, 3, -1, -1, 4, 5, 6, -1, -1, -1, 7, 8, 9, 10, 11, -1, 12, 13, -1, -1, -1, -1, 14, 15, 16, 17, 18, 19])
# index = torch.tensor([1, 2, 3, 4, 0, 0, 5, 6, 7, 0, 0, 0, 8, 9, 0, 0, 0, 0, 10, 11, 0, 0, 0, 0, 15, 16, 17, 18, 19, 20])

# output = torch.zeros(s)



# res = torch.index_add(output, dim=0, index=index, source=t.view(-1))

# print(res[1:])


# pkv_fast = torch.load("pkv_fast.pt", "cpu")
# pkv = torch.load("pkv.pt", "cpu")


# print(pkv_fast[0, 0, 0, 0])
# print(pkv[0, 0, 0, 0])


# inner_inputs = torch.load("inner_inputs.pt", map_location="cpu")

# print(inner_inputs["demo_input_ids"].size())
# print(inner_inputs["demo_position_ids"].size())
# print(inner_inputs["demo_attention_mask"].size())
# print(inner_inputs["demo_index"].size())
# print(inner_inputs["demo_index_inv"].size())

# print(inner_inputs["demo_input_ids"][0].tolist())
# print(inner_inputs["demo_position_ids"][0].tolist())
# # print(inner_inputs["demo_attention_mask"][0])
# # print(inner_inputs["demo_index"][0].tolist())
# print(inner_inputs["demo_index_inv"][13].tolist())



# demo_input_ids = inner_inputs["demo_input_ids"].view(-1)
# demo_position_ids = inner_inputs["demo_position_ids"].view(-1)
# # demo_index = inner_inputs["demo_index"].view(-1)
# demo_index_inv = inner_inputs["demo_index_inv"].view(-1)

# inputs = torch.index_select(demo_input_ids, dim=0, index=demo_index).view(2, -1)

# print(inputs.tolist())

# inputs = torch.index_select(demo_position_ids, dim=0, index=demo_index).view(2, -1)

# print(inputs.tolist())

# inputs = torch.index_select(demo_input_ids, dim=0, index=demo_index_inv)
# print(inputs.view(inner_inputs["demo_index_inv"].size()).tolist()[9])

# inputs = torch.index_select(demo_position_ids, dim=0, index=demo_index_inv)
# print(inputs.view(inner_inputs["demo_index_inv"].size()).tolist()[10])

# loss_func = nn.CrossEntropyLoss()

# mb_many, no_mb_many, icl_mb_many = torch.load("mb_many.pt", map_location="cpu")
# mb_few, no_mb_few = torch.load("mb_few.pt", map_location="cpu")

# logits_many = torch.load("logits_many.pt", map_location="cpu").squeeze()
# logits_few = torch.load("logits_few.pt", map_location="cpu").squeeze()

# label_many = no_mb_many["label"].squeeze()
# label_few = no_mb_few["label"].squeeze()

# input_ids_many = mb_many["input_ids"].squeeze()
# input_ids_few = mb_few["input_ids"].squeeze()


# print(logits_many.size())
# print(logits_few.size())


# loss_many = loss_func(logits_many.float().view(-1, logits_many.shape[-1]), no_mb_many["label"].view(-1))
# loss_few = loss_func(logits_few.float().view(-1, logits_few.shape[-1]), no_mb_few["label"].view(-1))

# print(loss_many.item())
# print(loss_few.item())

# index_many = torch.LongTensor([i for i in range(len(label_many)) if label_many[i] != -100])
# index_few = torch.LongTensor([i for i in range(len(label_few)) if label_few[i] != -100])

# print(index_many)
# print(index_few)

# print(input_ids_many)

# input_index_many = torch.LongTensor([i for i in range(len(input_ids_many)) if input_ids_many[i] != 50256])
# input_index_few = torch.LongTensor([i for i in range(len(input_ids_few)) if input_ids_few[i] != 50256])

# print(input_index_many)
# print(input_index_few)


# print(input_ids_many[:len(input_index_many)])
# tmp = input_ids_few[:len(input_index_few)]
# input_index_few_1 = input_index_few[:-len(input_index_many)]
# input_index_few_2 = input_index_few[-len(input_index_many):]


# logits_many_s = torch.index_select(logits_many, dim=0, index=index_many)
# logits_few_s = torch.index_select(logits_few, dim=0, index=index_few)

# print(logits_many_s)
# print(logits_few_s)

# print(torch.sum(torch.abs(logits_few_s - logits_many_s)))

# hidden_many = torch.load("hidden_many.pt", map_location="cpu").squeeze()
# hidden_few = torch.load("hidden_few.pt", map_location="cpu").squeeze()

# hidden_many_s = torch.index_select(hidden_many, dim=0, index=index_many)
# hidden_few_s = torch.index_select(hidden_few, dim=0, index=index_few)

# print(hidden_many_s)
# print(hidden_few_s)

# print(torch.sum(torch.abs(hidden_few_s - hidden_many_s)))


# print("qkv")

# for i in range(36):
#     demo_q_many, demo_k_many, demo_v_many = torch.load(f"demo_qkv_many_{i}.pt", map_location="cpu")
#     norm_q_few, norm_k_few, norm_v_few = torch.load(f"normal_qkv_few_{i}.pt", map_location="cpu")
#     print(list(map(len, icl_mb_many["input_ids"])))
#     print(demo_q_many.size())
#     print(norm_q_few.size())
    
#     s = 0
#     for bid, ids in enumerate(icl_mb_many["input_ids"]):
#         q_many = demo_q_many[bid][0][:len(ids)]
#         q_few = norm_q_few[0][0][s:s+len(ids)]
        
#         print(torch.sum(torch.abs(q_many - q_few)))
        
#         s += len(ids)
    
    # print(demo_k_many.size())
    # print(norm_k_few.size())
    # print(demo_v_many.size())
    # print(norm_v_few.size())

#     break

# print("qkv 2")

# for i in range(36):
#     demo_q_many, demo_k_many, demo_v_many = torch.load(f"demo_qkv_many_{i}_2.pt", map_location="cpu")
#     norm_q_few, norm_k_few, norm_v_few = torch.load(f"normal_qkv_few_{i}_2.pt", map_location="cpu")
#     print(list(map(len, icl_mb_many["input_ids"])))
#     print(demo_q_many.size())
#     print(norm_q_few.size())
    
#     s = 0
#     for bid, ids in enumerate(icl_mb_many["input_ids"]):
#         q_many = demo_q_many[bid][:len(ids)]
#         q_few = norm_q_few[0][s:s+len(ids)]
        
#         print(torch.sum(torch.abs(q_many - q_few)))
        
#         s += len(ids)
    
#     # print(demo_k_many.size())
#     # print(norm_k_few.size())
#     # print(demo_v_many.size())
#     # print(norm_v_few.size())

#     break



# print("attn")


# for i in range(36):
#     demo_attn_weights_many = torch.load(f"demo_attn_weights_many_{i}.pt", map_location="cpu")
#     norm_attn_weights_few = torch.load(f"normal_attn_weights_few_{i}.pt", map_location="cpu")
#     print(list(map(len, icl_mb_many["input_ids"])))
#     print(demo_attn_weights_many.size())
#     print(norm_attn_weights_few.size())
#     s = 0
#     for bid, ids in enumerate(icl_mb_many["input_ids"]):
#         demo_a = demo_attn_weights_many[bid][0][:len(ids), :len(ids)]
#         norm_a = norm_attn_weights_few[0][0][s:s+len(ids), s:s+len(ids)]
        
#         demo_a = torch.where(demo_a < -10000, 10, demo_a)
#         norm_a = torch.where(norm_a < -10000, 10, norm_a)
        
#         s += len(ids)
        
#         print(torch.sum(torch.abs(demo_a - norm_a)))
        
#     break



# for i in range(36):
#     hidden_many = torch.load(f"hidden_many_{i}.pt", map_location="cpu").squeeze()
#     hidden_few = torch.load(f"hidden_few_{i}.pt", map_location="cpu").squeeze()

#     hidden_many_s = torch.index_select(hidden_many, dim=0, index=index_many)
#     hidden_few_s = torch.index_select(hidden_few, dim=0, index=index_few)

#     # print(hidden_many_s)
#     # print(hidden_few_s)

#     print(torch.sum(torch.abs(hidden_few_s - hidden_many_s)))
    
    
# for i in range(36):
#     hidden_many = torch.load(f"inner_hidden_many_{i}.pt", map_location="cpu").squeeze()
#     hidden_few = torch.load(f"inner_hidden_few_{i}.pt", map_location="cpu").squeeze()

#     hidden_many_s = torch.index_select(hidden_many, dim=0, index=index_many)
#     hidden_few_s = torch.index_select(hidden_few, dim=0, index=index_few)

#     # print(hidden_many_s)
#     # print(hidden_few_s)

#     print(torch.sum(torch.abs(hidden_few_s - hidden_many_s)))


# print("hidden_before_attn")

# for i in range(36):
#     hidden_before_demo_many = torch.load(f"demo_hidden_before_attn_many_{i}.pt", map_location="cpu").squeeze()
#     hidden_before_demo_few = torch.load(f"hidden_before_attn_few_{i}.pt", map_location="cpu").squeeze()

#     # print(attn_output_few.size())
#     # print(input_index_few_2)
    
#     hidden_before_demo_many = torch.index_select(hidden_before_demo_many, dim=0, index=input_index_few_1)
#     hidden_before_demo_few = torch.index_select(hidden_before_demo_few, dim=0, index=input_index_few_1)
    
#     print(torch.sum(torch.abs(hidden_before_demo_many - hidden_before_demo_few)))

#     break


# print("hidden_before_attn 2")

# for i in range(36):
#     hidden_before_demo_many = torch.load(f"demo_hidden_before_attn_many_{i}_2.pt", map_location="cpu").squeeze()
#     hidden_before_demo_few = torch.load(f"hidden_before_attn_few_{i}.pt", map_location="cpu").squeeze()

#     print(hidden_before_demo_many.size())
#     print(hidden_before_demo_few.size())

#     s = 0
#     for bid, ids in enumerate(icl_mb_many["input_ids"]):
#         h_many = hidden_before_demo_many[bid][:len(ids)]
#         h_few = hidden_before_demo_few[s:s+len(ids)]

#         print(torch.sum(torch.abs(h_many - h_few)))

#         s += len(ids)

#     break

# print("my conv")

# class Conv1D(nn.Module):
#     """
#     1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

#     Basically works like a linear layer but the weights are transposed.

#     Args:
#         nf (`int`): The number of output features.
#         nx (`int`): The number of input features.
#     """

#     def __init__(self, nf, nx):
#         super().__init__()
#         self.nf = nf
#         w = torch.empty(nx, nf)
#         nn.init.normal_(w, std=0.02)
#         self.weight = nn.Parameter(w)
#         self.bias = nn.Parameter(torch.zeros(nf))

#     def forward(self, x):
#         size_out = x.size()[:-1] + (self.nf,)
#         x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
#         x = x.view(size_out)
#         return x

# conv = Conv1D(3 * 1280, 1280).cuda()

# demo_q_many, demo_k_many, demo_v_many = conv(hidden_before_demo_many.cuda()).split(1280, dim=2)
# norm_q_few, norm_k_few, norm_v_few = conv(hidden_before_demo_few.cuda()).split(1280, dim=1)
# print(list(map(len, icl_mb_many["input_ids"])))
# print(demo_q_many.size())
# print(norm_q_few.size())

# s = 0
# for bid, ids in enumerate(icl_mb_many["input_ids"]):
#     q_many = demo_q_many[bid][:len(ids)]
#     q_few = norm_q_few[s:s+len(ids)]
    
#     print(torch.sum(torch.abs(q_many - q_few)))
    
#     s += len(ids)

# print(demo_k_many.size())
# print(norm_k_few.size())
# print(demo_v_many.size())
# print(norm_v_few.size())




# print("demo attn before cproj")

# for i in range(36):
#     attn_output_many = torch.load(f"demo_attn_output_before_cproj_many_{i}.pt", map_location="cpu").squeeze()
#     attn_output_few = torch.load(f"attn_output_before_cproj_few_{i}.pt", map_location="cpu").squeeze()

#     # print(attn_output_few.size())
#     # print(input_index_few_2)
    
#     attn_output_many = torch.index_select(attn_output_many, dim=0, index=input_index_few_1)
#     attn_output_few = torch.index_select(attn_output_few, dim=0, index=input_index_few_1)
    
#     print(torch.sum(torch.abs(attn_output_many - attn_output_few)))
    
#     if i > 2:
        # break



# print("attn_output")    

# for i in range(36):
#     attn_output_many = torch.load(f"attn_output_many_{i}.pt", map_location="cpu").squeeze()
#     attn_output_few = torch.load(f"attn_output_few_{i}.pt", map_location="cpu").squeeze()

#     # print(attn_output_few.size())
#     # print(input_index_few_2)
    
#     attn_output_many = torch.index_select(attn_output_many, dim=0, index=input_index_many)
#     attn_output_few = torch.index_select(attn_output_few, dim=0, index=input_index_few_2)
    
#     print(torch.sum(torch.abs(attn_output_many - attn_output_few)))
    
#     if i > 2:
#         break

# print("demo attn")

# for i in range(36):
#     attn_output_many = torch.load(f"demo_attn_output_many_{i}.pt", map_location="cpu").squeeze()
#     attn_output_few = torch.load(f"attn_output_few_{i}.pt", map_location="cpu").squeeze()

#     # print(attn_output_few.size())
#     # print(input_index_few_2)
    
#     attn_output_many = torch.index_select(attn_output_many, dim=0, index=input_index_few_1)
#     attn_output_few = torch.index_select(attn_output_few, dim=0, index=input_index_few_1)
    
#     print(torch.sum(torch.abs(attn_output_many - attn_output_few)))
    
#     if i > 2:
#         break

# from datasets import load_dataset

# data_files = {
#     "validation": "/home/guyuxian/data_hf/mrpc/cache/validation.jsonl",
# }
# dataset = load_dataset("json", data_files=data_files)

# dataset = dataset.shuffle(seed=42)

# print(dataset["validation"][:5])

# dataset = dataset.shuffle(seed=42)

# print(dataset["validation"][:5])

# from transformers import GPT2Tokenizer
# import pickle

# tokenizer = GPT2Tokenizer.from_pretrained("/home/guyuxian/CodeRepo/icl_train/results_new_new/gpt2-large")

# with open("/home/guyuxian/data_hf/mrpc/cache/icl_cache_train_1_10000_1024.pkl", "rb") as f:
#     past_train_cache = pickle.load(f)

# # print(past_train_cache[0]["mrpc"]["equivalent"][0])
# print(tokenizer.decode(past_train_cache[0]["mrpc"]["equivalent"][0]["context_ids"]))
# print("=" * 100)
# # print(past_train_cache[0]["mrpc"]["same thing"][0])
# print(tokenizer.decode(past_train_cache[0]["mrpc"]["same thing"][0]["context_ids"]))

# with open("/home/guyuxian/data_hf/mrpc/cache/icl_new_cache_train_1_10000_1024_42_equivalent.pkl", "rb") as f:
#     now_train_eq = pickle.load(f)
    
# with open("/home/guyuxian/data_hf/mrpc/cache/icl_new_cache_train_1_10000_1024_42_same thing.pkl", "rb") as f:
#     now_train_same = pickle.load(f)
# print("*" * 100)
# print(tokenizer.decode(now_train_eq[0]["context_ids"]))
# print("=" * 100)
# print(tokenizer.decode(now_train_same[0]["context_ids"]))

# too_long = torch.load("too_long.pt", map_location="cpu")

# print(too_long.tolist())
# print(len(too_long))

# tag = "in_model"

model_batch = torch.load(f"model_batch.pt", map_location="cpu")
input_ids = model_batch["input_ids"]

icl_batch = torch.load(f"icl_batch.pt", map_location="cpu")

inner_inputs = torch.load(f"inner_inputs.pt", map_location="cpu")

print(model_batch["position_ids"][1])
print(inner_inputs["demo_position_ids"][1].tolist())
# test_all_demo_ids = torch.load("/home/guyuxian/many_coreset/inner_res/test_demo_ids.pt", map_location="cpu")

# print(test_all_demo_ids)

# inner_no_model_batch = inner_inputs[0][1]
# print(inner_no_model_batch["valid_lengths"])
# print(inner_no_model_batch["demo_start_pos"])

# l = 11

# hidden = torch.load(f"hidden_{tag}_{l}.pt", map_location="cpu")
# print(hidden)

# demo_input_ids = inner_inputs["demo_position_ids"]
# print(demo_input_ids)
# print(inner_inputs["demo_index_inv"])
# print(inner_inputs["demo_index_inv"].size())

# x = torch.index_select(demo_input_ids.view(-1), dim=0, index=inner_inputs["demo_index_inv"].view(-1))

# x = x.view(inner_inputs["demo_index_inv"].size(0), inner_inputs["demo_index_inv"].size(1))

# print(x[0].tolist())

# x = torch.index_select(x.view(-1), dim=0, index=inner_inputs["demo_context_index"].view(-1))

# x = x.view(*inner_inputs["demo_context_index"].size())

# print(x[0].tolist())