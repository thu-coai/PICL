port=2040
base_dir=/home/lidong1/CodeRepo/icl_train/scripts/infer_debug

for seed in 10 20 30 40 50
do
shot=4
ckpt=meta_icl/train/MCQA-EXQA-CBQA-S2T-PARA-HR_NEW/10K/vanilla/test_target/chunk-1/pos0_%253Cn%253E/shot16/lr1e-06-bs1-G1-N8-wm0/len256-1024-None/gpt2-xl/10-10-42/30000
bash ${base_dir}/infer_icl_vanilla_debug_new.sh /home/lidong1/CodeRepo $port 1 $seed 10 $shot $ckpt --eval-batch-size 8
done

for seed in 10 20 30 40 50
do
shot=8
ckpt=meta_icl/train/MCQA-EXQA-CBQA-S2T-PARA-HR_NEW/10K/vanilla/test_target/chunk-1/pos0_%253Cn%253E/shot16/lr1e-06-bs1-G1-N8-wm0/len256-1024-None/gpt2-xl/10-10-42/30000
bash ${base_dir}/infer_icl_vanilla_debug_new_1.sh /home/lidong1/CodeRepo $port 1 $seed 10 $shot $ckpt --eval-batch-size 8
done