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
# CKPT_NAME="meta_icl/train/HR/vanilla/test_target/chunk-1/pos0_<n>_r2s_trim/shot16/lr1e-05-bs1-G1-N8/len256-1024-None/gpt2-large/10-10-42/30000/"
CKPT="${BASE_PATH}/icl_train/results/${CKPT_NAME}/"
# data
DATA_PREFIX="${4-"general/10M_256/256/roberta-base/HR/pos1_easy_neg1_hard_neg1_seed42_concate32/bs64_32_lr0.00005_G1_SEED/4000.pt/l2_h5/res_-1/r2s<n>/"}"
LM_DATA_PREFIX="${5-"lm_data/1024/no_stuffed_valid100K/rr<n>/"}"
DATA_DIR="${BASE_PATH}/icl_train/unsup_data/${DATA_PREFIX}"
LM_DATA_DIR="${BASE_PATH}/icl_train/unsup_data_test/${LM_DATA_PREFIX}"
# hp
EVAL_BATCH_SIZE=32
# length
MAX_LENGTH=1024
MAX_LENGTH_ALL_DEMOS=-1
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/icl_train/unsup_data_test/${LM_DATA_PREFIX}/filter_res/"
# seed
SEED=10
SEED_ORDER=10


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
# OPTS+=" --lm-data-name filter_lm_10K"
OPTS+=" --lm-ratio 100"
OPTS+=" --num-workers 4"
# OPTS+=" --optim-batch"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE}"
OPTS+=" --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS}"
# runtime
OPTS+=" --log-interval 1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --no-extend-save-path"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/icl_train/configs/deepspeed/ds_config.json"
# icl
OPTS+=" --icl-sup all_target"


export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/filter.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
