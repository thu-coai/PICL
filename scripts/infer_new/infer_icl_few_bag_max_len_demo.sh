#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2118}
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
CKPT_NAME=${6-"gpt2-large"}
CKPT="${BASE_PATH}/icl_train/results_cache/${CKPT_NAME}/"
SEED=${3-10}
SEED_ORDER=${4-10}
TYPE="few_bag"
POS_TYPE=2
EVAL_BATCH_SIZE=8
MAX_LENGTH=-1
MAX_LENGTH_ALL_DEMOS=40960
MAX_LENGTH_PER_SAMPLE=256
SHOT=${5-16}

SAVE_PATH="${BASE_PATH}/icl_train/results_cache/infer/${DATA_NAMES}/${TYPE}/${SHOT}/${CKPT_NAME}/${SEED}-${SEED_ORDER}/ml${MAX_LENGTH_PER_SAMPLE}_${MAX_LENGTH}_${MAX_LENGTH_ALL_DEMOS}/share-balance/"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE}"
OPTS+=" --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 1000"
OPTS+=" --do-eval"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --shot ${SHOT}"
OPTS+=" --icl-share-demo"
OPTS+=" --icl-balance"
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
OPTS+=" --train-prompts Simple_Prompt"
OPTS+=" --eval-prompts Simple_Prompt"
OPTS+=" --type ${TYPE}"
OPTS+=" --pos-type ${POS_TYPE}"
OPTS+=" --icl-bag-of-inst"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/icl_train/configs/deepspeed/ds_config.json"
# OPTS+=" --force-process"
# OPTS+=" --force-process-demo"
OPTS+=" --data-process-workers -1"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/finetune_gpt2_few_ds.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
# ${CMD} 2>&1 | tee ${SAVE_PATH}/train.log
