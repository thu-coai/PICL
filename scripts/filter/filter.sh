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
BASE_PATH=${1}
CKPT_NAME="gpt2-large"
CKPT="${BASE_PATH}/results/${CKPT_NAME}/"
# data
RAW_DATA="100K_128"
SEARCH_DATA="${RAW_DATA}/TRAIN_p1_en1_hn4_s42_lr5e-05-bs64-G1_4000.pt/L2"
DATA_DIR="${BASE_PATH}/pretrain_data/raw/${RAW_DATA}"
IDX_DATA_DIR="${BASE_PATH}/pretrain_data/retrieval_results/${SEARCH_DATA}"
# hp
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/pretrain_data/filter_results/"
# seed
SEED=10
SEED_ORDER=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-dir ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --picl-data-dir ${DATA_DIR}"
OPTS+=" --picl-idx-data-dir ${IDX_DATA_DIR}"
OPTS+=" --picl-data-prefix ${SEARCH_DATA}"
OPTS+=" --picl-data-name picl"
OPTS+=" --num-workers 4"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE}"
# runtime
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# icl
OPTS+=" --icl-sup all_target"
# filter
OPTS+=" --score-zero"
OPTS+=" --score-icl"
OPTS+=" --do-filter"
OPTS+=" --filter-num 10000"
OPTS+=" --filter-threshold 0.0"


export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/filter.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"

mkdir -p ${SAVE_PATH}

CODE_BASE=HF ${CMD}
