#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

BASE_PATH=${1-"/home/guyuxian/CodeRepo"}
DATA_NAMES="sst2-rte-cb-copa-wsc-wic"
DATA_DIR="${BASE_PATH}/data/"
CKPT_NAME="MCQA-EXQA-CBQA-TC-PARA/test_target/16/gpt2-large/bmt/10-10/lr0.00005_bs64_G1_ml512/3200"
CKPT="${BASE_PATH}/icl_train/results/train/${CKPT_NAME}"
SEED=${3-10}
SEED_ORDER=${4-10}
TYPE="few"
BATCH_SIZE=64
SHOT=${5-8}

SAVE_PATH="${BASE_PATH}/icl_train/results_new/infer/${DATA_NAMES}/${TYPE}/${SHOT}/${CKPT_NAME}/${SEED}-${SEED_ORDER}/share-balance/"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --train-iters 400"
OPTS+=" --max-length 1024"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
OPTS+=" --do-eval"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --shot ${SHOT}"
OPTS+=" --icl-share-train-data"
OPTS+=" --icl-balance"
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
OPTS+=" --eval-prompts Simple_Prompt"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/finetune_gpt2_few.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
# ${CMD} 2>&1 | tee ${SAVE_PATH}/train.log
