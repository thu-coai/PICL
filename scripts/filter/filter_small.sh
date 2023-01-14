#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2031}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"/home/lidong1/CodeRepo"}
CKPT_NAME=${4-"meta_icl/train/HR/vanilla/all_full/chunk-1/pos0_<n>_r2s_trim/shot16/lr1e-05-bs1-G1-N8/len256-1024-None/gpt2-large/10-10-42/30000"}
CKPT="${BASE_PATH}/icl_train/results/${CKPT_NAME}/"
# data
DATA_DIR="/home/lidong1/CodeRepo/icl_train/paws_origin-labeled_final_Concatenation.jsonl"
# hp
EVAL_BATCH_SIZE=32
# length
MAX_LENGTH=1024
MAX_LENGTH_ALL_DEMOS=-1
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/icl_train/unsup_data/small_filter/"
# seed
SEED=10
SEED_ORDER=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
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
OPTS+=" --score-small"


export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/filter.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
