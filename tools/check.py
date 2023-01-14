import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("/home/guyuxian/checkpoints/gpt2-large/bmt/")

# model_batch = torch.load("model_batch_new.pt", map_location="cpu")
# no_model_batch = torch.load("no_model_batch_new.pt", map_location="cpu")

# idx = 2

# print(model_batch["input_ids"][idx].tolist())
# print(tokenizer.decode(model_batch["input_ids"][idx].tolist()))
# print(model_batch["attention_mask"][idx])
# print(model_batch["position_ids"][idx])

# print(no_model_batch["idxs"][idx])
# print(no_model_batch["label"][idx])
# print(no_model_batch["loss_mask"][idx])

# mask = (no_model_batch["label"][idx] != -100)
# s_in = torch.masked_select(model_batch["input_ids"][idx], mask)
# s_la = torch.masked_select(no_model_batch["label"][idx], mask)
# s_m = torch.masked_select(no_model_batch["loss_mask"][idx], mask)
# print(s_in)
# print(s_la)
# print(s_m)

# print(tokenizer.decode(s_in))
# print(tokenizer.decode(s_la))


# model_batch = torch.load("model_batch_dev.pt", map_location="cpu")
# no_model_batch = torch.load("no_model_batch_dev.pt", map_location="cpu")

# idx = 0

# print(model_batch["input_ids"][idx].tolist())
# print(tokenizer.decode(model_batch["input_ids"][idx].tolist()))
# print(model_batch["attention_mask"][idx])
# print(model_batch["position_ids"][idx])

# print(no_model_batch["idxs"][idx])
# print(no_model_batch["label"][idx])
# print(no_model_batch["loss_mask"][idx])
# print(no_model_batch["pos_mask"][idx])
# print(no_model_batch["option_label"][idx])

# mask = (no_model_batch["label"][idx] != -100)
# s_in = torch.masked_select(model_batch["input_ids"][idx], mask)
# s_la = torch.masked_select(no_model_batch["label"][idx], mask)
# s_m = torch.masked_select(no_model_batch["loss_mask"][idx], mask)

# print(s_in)
# print(s_la)
# print(s_m)

# print(tokenizer.decode(s_in))
# print(tokenizer.decode(s_la))

# for x in model_batch["attention_mask"][idx].int().tolist():
#     print(x)

# input_ids_lm = torch.load("input_ids_lm.pt", map_location="cpu")
# input_ids_test = torch.load("input_ids_test.pt", map_location="cpu")

# print(input_ids_lm.size())
# print(input_ids_test.size())

# print(torch.abs(torch.sum(input_ids_test - input_ids_lm)))

# print(input_ids_test)
# print(input_ids_lm)

# large = torch.load("/home/guyuxian/checkpoints/gpt2-large/pytorch_model.bin", map_location="cpu")
# print(large["wte.weight"].size())

# large = torch.load("/home/guyuxian/checkpoints/gpt2-large/bmt/pytorch_model.pt", map_location="cpu")
# print(large["input_embedding.weight"].size())

# xl = torch.load("/home/guyuxian/checkpoints/gpt2-xl/pytorch_model.bin", map_location="cpu")
# print(xl["wte.weight"].size())

# xl = torch.load("/home/guyuxian/checkpoints/gpt2-xl/bmt/pytorch_model.pt", map_location="cpu")
# print(xl["input_embedding.weight"].size())

# kv_hf = torch.load("/home/guyuxian/CodeRepo/icl/kv_hf.pt", map_location="cpu")
# kv_bm = torch.load("/home/guyuxian/CodeRepo/icl_train/kv_bm.pt", map_location="cpu")


# print(kv_hf.size())
# print(kv_bm.size())

# print(kv_hf[0, 0, 0, 0])
# print(kv_bm[0, 0, 0, 0])


# bm_hf = torch.load("/home/guyuxian/CodeRepo/icl/mb_hf.pt", map_location="cpu")
# bm_mb = torch.load("/home/guyuxian/CodeRepo/icl_train/mb_bm.pt", map_location="cpu")

# print(bm_hf["attention_mask"][0])
# print(bm_mb["attention_mask"][0])

# pkv_iter_hf, length_hf = torch.load("/home/guyuxian/CodeRepo/icl/pkv_iter_hf.pt", map_location="cpu")
# pkv_iter_bm, length_bm = torch.load("/home/guyuxian/CodeRepo/icl_train/pkv_iter_bm.pt", map_location="cpu")

# print(length_hf)
# print(length_bm)

# print(-length_hf[0], length_bm[0])
# print(pkv_iter_hf[0, 0, 0, 0, -length_hf[0]:])
# print(pkv_iter_bm[0, 0, 0, 0, :length_bm[0]])

# idx = 1

# train_mb = torch.load("/home/guyuxian/CodeRepo/icl_train/un_model_batch.pt", map_location="cpu")
# train_no_mb = torch.load("/home/guyuxian/CodeRepo/icl_train/un_no_model_batch.pt", map_location="cpu")
# icl_b = torch.load("/home/guyuxian/CodeRepo/icl_train/un_icl_batch.pt", map_location="cpu")

# print(train_mb["input_ids"][idx])
# # print(len(train_mb["input_ids"][idx]))
# print(train_mb["position_ids"][idx])
# print(tokenizer.decode(train_mb["input_ids"][idx]))
# # print(train_no_mb["label"][idx])
# # print(tokenizer.decode(train_no_mb["label"][idx]))

# # for ids, pos in zip(icl_b["input_ids"], icl_b["pos"]):
# #     if pos[0] == idx:
# #         print(tokenizer.decode(ids))
# #         print("*" * 100)

# inner_mb = torch.load("/home/guyuxian/CodeRepo/icl_train/inner_mb.pt", map_location="cpu")
# inner_no_mb = torch.load("/home/guyuxian/CodeRepo/icl_train/inner_no_mb.pt", map_location="cpu")

# # print(tokenizer.decode(inner_mb["input_ids"][0]))
# print(len(inner_mb["input_ids"]))
# print(len(inner_mb["position_ids"]))
# print(inner_no_mb["pos"])
# for ids, pids, pos in zip(inner_mb["input_ids"], inner_mb["position_ids"], inner_no_mb["pos"]):
#     if pos[0] == idx:
#         print(pids)


idx = 0

mb = torch.load("/home/guyuxian/CodeRepo/icl_train/mb.pt", map_location="cpu")
no_mb = torch.load("/home/guyuxian/CodeRepo/icl_train/no_mb.pt", map_location="cpu")

print(tokenizer.decode(mb["input_ids"][idx]))
print(no_mb["label"][idx].tolist())

xx = []
yy = []

for x, y in zip(mb["input_ids"][idx].tolist(), no_mb["label"][idx].tolist()):
    if y != -100:
        xx.append(x)
        yy.append(y)

print(xx)
print(yy)

print(mb["position_ids"][idx].tolist())