PATH_TO_CKPT="results/picl-large" # path to your ckpt in results/
BASE_PATH=${1}

for seed in 10 20 30 40 50
do
    bash scripts/eval/eval_cls.sh ${BASE_PATH} 2113 1 $seed 10 4 ${PATH_TO_CKPT}
done