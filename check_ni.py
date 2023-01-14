import json
import os

# res_vanilla = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni_fix/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/meta_icl_train_MCQA-EXQA-CBQA-S2T-PARA-HR_NEW_10K_vanilla_test_target_chunk-1_pos0_%253Cn%253E_shot16_lr1e-05-bs4-G1-N8-wm0_len256-256-None_gpt2-large_10-10-42_30000/10-10-42/res.json"
# res_ours = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni_fix/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/res.json"


# with open(res_vanilla) as f:
#     res_vanilla = json.load(f)
    
# with open(res_ours) as f:
#     res_ours = json.load(f)
    
# n = 0
# nl = 0
# for k in res_vanilla:
#     if "rougeL" in k:
#         n += 1
#         if res_vanilla[k] > res_ours[k]:
#             nl += 1
#             print(round(res_vanilla[k]-res_ours[k], 4), "\t\t", k, res_vanilla[k], res_ours[k])
        
# print(n-12)
# print(nl-3)
        
from ni_tasks import ALL_NI_TASKS
from collections import defaultdict


cat = defaultdict(list)

for task in ALL_NI_TASKS:
    with open(os.path.join("/home/lidong1/CodeRepo/nat_inst/natural-instructions/tasks/", f"{task}.json")) as f:
        data = json.load(f)
        cat[data["Categories"][0]].append(task)

n = 0

for k in cat:
    print(k)
    for i in range(0, len(cat[k]), 2):
        if i + 1 < len(cat[k]):
            print("& " + cat[k][i].replace("_", "\\_") + " & " + cat[k][i+1].replace("_", "\\_") + " \\\\")
            n += 2
        else:
            print("& " + cat[k][i].replace("_", "\\_") + " & " +" \\\\")
            n += 1
    print()
    
print(n)



from ni_tasks import EXCLUDE_TASKS
from collections import defaultdict

n = 0
for i in range(0, len(EXCLUDE_TASKS), 2):
    if i + 1 < len(EXCLUDE_TASKS):
        print("& " + EXCLUDE_TASKS[i].replace("_", "\\_") + " & " + EXCLUDE_TASKS[i+1].replace("_", "\\_") + " \\\\")
        n += 2
    else:
        print("& " + EXCLUDE_TASKS[i].replace("_", "\\_") + " & " +" \\\\")
        n += 1
    
print(n)


# path = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/gpt2-large/10-10-42/res.json"
# path = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/pretrain_lm_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_only_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-05-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_60000/10-10-42/res.json"
# path = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/pretrain_mixed_general_full_256_rand_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/res.json"
# path = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/res.json"

# path = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/pretrain_general_full_256_selfsup_15M_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/res.json"

# path = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/meta_icl_train_MCQA-EXQA-CBQA-S2T-PARA-HR_NEW_10K_vanilla_test_target_chunk-1_pos0_%253Cn%253E_shot16_lr1e-05-bs4-G1-N8-wm0_len256-256-None_gpt2-large_10-10-42_30000/10-10-42/res.json"

# path = "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_ni/NI/balance_eval/vanilla/test_target/chunk-1/pos0_nn_r2s_trim/shot0/len256-1024-None/gpt2-xl/10-10-42/res.json"

# with open(path) as f:
#     res = json.load(f)
    
    
# for k in res:
#     if "rougeL" in k and "task" not in k:
#         print(k, res[k])