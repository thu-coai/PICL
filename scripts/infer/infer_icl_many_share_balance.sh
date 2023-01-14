#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2015}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=${1-"/home/guyuxian/CodeRepo"}
DATA_NAMES="sst2"
DATA_DIR="${BASE_PATH}/data/"
CKPT_NAME="gpt2-large/bmt/"
CKPT="/home/guyuxian/checkpoints/${CKPT_NAME}"
SEED=${3-10}
SEED_ORDER=${4-10}
TYPE="many"
POS_TYPE=2
EVAL_BATCH_SIZE=32
INNER_BATCH_SIZE=128
MAX_LENGTH=256
ICL_TRAIN_MAX_LENGTH=40960
SHOT=${5-64}

SAVE_PATH="${BASE_PATH}/icl_train/results_new/infer/${DATA_NAMES}/${TYPE}_${POS_TYPE}_bos/${SHOT}/${CKPT_NAME}/${SEED}-${SEED_ORDER}/share-balance/"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --train-iters 400"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --icl-train-max-length ${ICL_TRAIN_MAX_LENGTH}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --num-workers 1"
OPTS+=" --dev-num 1000"
OPTS+=" --do-eval"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --icl-inner-batch-size ${INNER_BATCH_SIZE}"
OPTS+=" --shot ${SHOT}"
OPTS+=" --icl-share-train-data"
OPTS+=" --icl-balance"
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
OPTS+=" --eval-prompts Simple_Prompt"
OPTS+=" --unordered-pos-type ${POS_TYPE}"
OPTS+=" --add-bos"
# OPTS+=" --icl-demo-len 34"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/finetune_gpt2_many.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
# ${CMD} 2>&1 | tee ${SAVE_PATH}/train.log
