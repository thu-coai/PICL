PATH_TO_CKPT="xxx" # path to your ckpt in results/

for seed in 10 20 30 40 50
do
    bash scripts/eval/eval_cls.sh ${BASE_PATH} 2113 1 $seed 10 4 ${PATH_TO_CKPT}
done