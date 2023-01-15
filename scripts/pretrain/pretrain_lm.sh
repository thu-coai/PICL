#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-2}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/guyuxian/CodeRepo"}
CKPT_NAME="gpt2-large"
CKPT="${BASE_PATH}/results/${CKPT_NAME}/"
# data
DATA_PREFIX="${4-"general/10M_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/l2_h5/res_-1/r2s<n>/"}"
LM_DATA_PREFIX="${5-"lm_data/1024/stuffed/rr<n>/"}"
DATA_DIR="${BASE_PATH}/pretrain_data/${DATA_PREFIX}"
LM_DATA_DIR="${BASE_PATH}/pretrain_data/${LM_DATA_PREFIX}"
# hp
BATCH_SIZE=2
LR=0.00001
LR_DECAY="noam"
GRAD_ACC=16
# length
MAX_LENGTH=1024
MAX_LENGTH_ALL_DEMOS=-1
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/results/pretrain/"
# seed
SEED=10
SEED_ORDER=10
# icl
TYPE="vanilla"
ICL_SUP=all_target
EVAL_BATCH_SIZE=32
SHOT=16


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --unsup-data-name tokenized"
OPTS+=" --unsup-data-prefix ${DATA_PREFIX}"
OPTS+=" --lm-data-prefix ${LM_DATA_PREFIX}"
OPTS+=" --lm-ratio 1"
OPTS+=" --lm-only"
OPTS+=" --num-workers 4"
OPTS+=" --dev-ratio 0.1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style ${LR_DECAY}"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 2048"
OPTS+=" --epochs 1"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE}"
OPTS+=" --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS}"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
# OPTS+=" --do-eval"
OPTS+=" --save-interval 1000"
OPTS+=" --eval-interval 1000"
OPTS+=" --log-interval 1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/icl_train/configs/deepspeed/ds_config.json"
# icl
OPTS+=" --type ${TYPE}"
OPTS+=" --shot ${SHOT}"
OPTS+=" --icl-sup ${ICL_SUP}"
# OPTS+=" --attn-dtype float"


export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/pretrain_gpt2_few_ds.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
