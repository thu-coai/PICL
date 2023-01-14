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
BASE_PATH=${1-"/home/guyuxian/CodeRepo"}
CKPT_NAME=${7-"gpt-neo"}
CKPT="${BASE_PATH}/icl_train/results/${CKPT_NAME}/"
# data
DATA_NAMES="gigaword_eval-squad_qg_eval-story_gen_eval_2"
DATA_DIR="${BASE_PATH}/data/"
# hp
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=1024
MAX_LENGTH_ALL_DEMOS=-1
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/icl_train/results/meta_icl/infer_debug/"
# seed
SEED=${4-10}
SEED_ORDER=${5-10}
# icl
TYPE="vanilla"
SHOT=${6-16}


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --prompt-type origin"
OPTS+=" --num-workers 2"
OPTS+=" --dev-num 100"
OPTS+=" --balance-eval"
# OPTS+=" --force-process"
# OPTS+=" --force-process-demo"
OPTS+=" --data-process-workers -1"
OPTS+=" --trim"
OPTS+=" --replace-return-with-space"
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
OPTS+=" --reset-seed-each-data"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/icl_train/configs/deepspeed/ds_config.json"
# icl
OPTS+=" --shot ${SHOT}"
OPTS+=" --icl-share-demo"
OPTS+=" --icl-balance"
OPTS+=" --type ${TYPE}"
# OPTS+=" --train-prompts Simple_Prompt_2"
# OPTS+=" --eval-prompts Simple_Prompt_2"

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/finetune_gpt_neo_few_ds.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
