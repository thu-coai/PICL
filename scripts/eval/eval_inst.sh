#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1}
CKPT_NAME=${4-"gpt2-large"}
CKPT="${BASE_PATH}/results/${CKPT_NAME}/"
# data
DATA_NAMES="NI"
DATA_DIR="${BASE_PATH}/data/sni_eval/"
# hp
EVAL_BATCH_SIZE=4
# length
MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/results/eval/sni"
# seed
SEED=10
SEED_ORDER=10
# icl
SHOT=0 # test shots depend on each task in sni


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-dir ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --num-workers 2"
OPTS+=" --balance-eval"
OPTS+=" --sni-ref-file ${DATA_DIR}/test_references.jsonl"
# OPTS+=" --force-process"
# OPTS+=" --force-process-demo"
OPTS+=" --data-process-workers -1"
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
OPTS+=" --reset-seed-each-data"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# icl
OPTS+=" --shot ${SHOT}"

export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/evaluate_inst.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
