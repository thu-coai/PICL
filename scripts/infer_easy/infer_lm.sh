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
CKPT="${BASE_PATH}/icl_train/results/${CKPT_NAME}/"
# data
LM_DATA_PREFIX="${5-"lm_data/1024/no_stuffed_valid100K/rr<n>/"}"
LM_DATA_DIR="${BASE_PATH}/icl_train/unsup_data_test/${LM_DATA_PREFIX}"
# hp
EVAL_BATCH_SIZE=32
# length
MAX_LENGTH=1024
MAX_LENGTH_ALL_DEMOS=-1
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/icl_train/results/infer_lm/"
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
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --lm-data-prefix ${LM_DATA_PREFIX}"
OPTS+=" --num-workers 4"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE}"
OPTS+=" --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS}"
# runtime
OPTS+=" --do-eval"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/icl_train/configs/deepspeed/ds_config.json"

OPTS+=" --type lm"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/compute_lm.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
