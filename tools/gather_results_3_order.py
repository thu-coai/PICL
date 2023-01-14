import re
import json
from collections import defaultdict

file_names = [
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_filter_score1.5_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_50000/10-10-42/log.txt",    
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_filter_score1.5_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_60000/10-10-42/log.txt",    
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_filter_score1.5_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",    


    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SROBERTA_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_50000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SROBERTA_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_60000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SROBERTA_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",

    
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_filter_score1.0_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_50000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_filter_score1.0_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_60000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_filter_score1.0_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot5/len256-1024-None/meta_icl_train_MCQA-EXQA-CBQA-S2T-PARA-HR_NEW_10K_vanilla_test_target_chunk-1_pos0_%253Cn%253E_shot16_lr1e-05-bs4-G1-N8-wm0_len256-256-None_gpt2-large_10-10-42_30000/10-10-42/log.txt",

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/piqa_eval-hellaswag_eval-copa_eval-arc_easy_eval-arc_hard_eval-commonsense_qa_eval-openbookqa_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/gpt2-xl/10-10-42/log.txt"

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/gpt-neo/10-10-42/log.txt"

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/piqa_eval-hellaswag_eval-copa_eval-arc_easy_eval-arc_hard_eval-commonsense_qa_eval-openbookqa_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/gpt-neo/10-10-42/log.txt"

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_60000/10-10-42/log.txt",

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/piqa_eval-hellaswag_eval-copa_eval-arc_easy_eval-arc_hard_eval-commonsense_qa_eval-openbookqa_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_50000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/piqa_eval-hellaswag_eval-copa_eval-arc_easy_eval-arc_hard_eval-commonsense_qa_eval-openbookqa_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_60000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/piqa_eval-hellaswag_eval-copa_eval-arc_easy_eval-arc_hard_eval-commonsense_qa_eval-openbookqa_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot10/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot15/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_filtered_ppl0.1_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_new_50000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_filtered_ppl0.1_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_new_60000/10-10-42/log.txt",    
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_filtered_ppl0.1_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_new_70000/10-10-42/log.txt",    

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot5/len256-1024-None/pretrain_mixed_general_full_256_SROBERTA_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_50000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot5/len256-1024-None/pretrain_mixed_general_full_256_SROBERTA_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_60000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot5/len256-1024-None/pretrain_mixed_general_full_256_SROBERTA_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot5/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_0.10.1_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot5/len256-1024-None/pretrain_mixed_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_10.010.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt"

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/pretrain_mixed_general_full_60_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst5_eval/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot15/len256-1024-None/pretrain_mixed_general_full_60_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt"
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/gpt-neo/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot8/len256-1024-None/gpt-neo/10-10-42/log.txt",

    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug_ub/sst5_eval-ag_news_eval_6/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/gpt2-large/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug_ub/sst5_eval-ag_news_eval_6/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/gpt2-xl/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug_ub/sst5_eval-ag_news_eval_6/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/gpt-neo/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug_ub/sst5_eval-ag_news_eval_6/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/pretrain_lm_general_full_256_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_only_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-05-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug_ub/sst5_eval-ag_news_eval_6/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/pretrain_general_full_256_selfsup_15M_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm0_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug_ub/sst5_eval-ag_news_eval_6/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/meta_icl_train_MCQA-EXQA-CBQA-S2T-PARA-HR_NEW_10K_vanilla_test_target_chunk-1_pos0_%253Cn%253E_shot16_lr1e-05-bs4-G1-N8-wm0_len256-256-None_gpt2-large_10-10-42_30000/10-10-42/log.txt",
    # "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug_ub/sst5_eval-ag_news_eval_6/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot2/len256-1024-None/pretrain_mixed_general_full_60_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",

    "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/gpt2-large/10-10-42/log.txt",
    "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_100_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt",
    "/home/lidong1/CodeRepo/icl_train/results/meta_icl/infer_debug/sst2_eval-subj_eval_4-mr_eval_2-rte_eval-ag_news_eval_6-cb_eval_3/balance_eval/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot4/len256-1024-None/pretrain_mixed_general_full_60_SEARCH1_r2s_15M_lm_data_full_stuffed_rr_1.01.0_vanilla_all_target_chunk-1_pos0_nn_shot16_lr1e-06-bs1-G8-N16-wm1000_len256-1024-None_gpt2-large_10-10-42_70000/10-10-42/log.txt"
]

all_file_names = []

for name in file_names:
    for seed in [10, 20, 30, 40, 50]:
        all_file_names.append(name.replace("10-10-42/log.txt", f"10-{seed}-42/log.txt"))
# all_file_names = file_names

for i, name in enumerate(all_file_names):
    # print(name)
    with open(name) as f:
        lines = f.readlines()
        
    lines = [line.strip() for line in lines if len(line.strip()) > 0]

    p = re.compile(r"test \| name: (.*) \| avg res: (.*) \| avg loss: (.*) \| avg tot loss: (.*) \| res: ({.*}) \| loss: ({.*}) \| tot loss: ({.*})")

    res = []

    for line in lines:
        m = p.match(line)
        if m is not None:
            # print(m.group(1))
            # print(m.group(2))
            # print(m.group(3))
            all_res = json.loads(m.group(5).replace("\'", "\""))
            v = list(all_res.values())[0]
            res.append(str(round(v.get("accuracy", v.get("bleu", 0)), 2)))
    if i != 0 and i % 5 == 0:
        print()
        print()
    print("\t".join(res))
