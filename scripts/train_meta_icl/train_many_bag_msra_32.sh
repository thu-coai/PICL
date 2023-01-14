#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2015}
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
DATA_NAMES="HR"
DATA_DIR="${BASE_PATH}/data/"
# hp
BATCH_SIZE=1
LR=0.00001
GRAD_ACC=4
# length
MAX_LENGTH=-1
MAX_LENGTH_ALL_DEMOS=2048
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/icl_train/results/meta_icl/train/"
# seed
SEED=${3-10}
SEED_ORDER=${4-10}
# icl
TYPE="many_bag"
CHUNK_LEN=1024
POS_TYPE=3
ICL_SUP="test_target"
SHOT=32


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --prompt-type origin"
OPTS+=" --num-workers 2"
OPTS+=" --train-num 10000"
OPTS+=" --dev-num 100"
# OPTS+=" --force-process"
# OPTS+=" --force-process-demo"
OPTS+=" --data-process-workers -1"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --warmup-iters 0"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 2048"
OPTS+=" --epochs 10"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE}"
OPTS+=" --max-length-all-demos ${MAX_LENGTH_ALL_DEMOS}"
# runtime
OPTS+=" --do-train"
OPTS+=" --save-interval 2000"
OPTS+=" --eval-interval 10000000000000"
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
OPTS+=" --pos-type ${POS_TYPE}"
OPTS+=" --add-bos"
OPTS+=" --remove-inner-bos"
OPTS+=" --attn-scale"
# OPTS+=" --chunk-len ${CHUNK_LEN}"
OPTS+=" --icl-many-in-model"
OPTS+=" --attn-dtype float"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/finetune_gpt2_many_ds.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
