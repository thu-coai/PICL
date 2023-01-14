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
DATA_NAMES="MCQA-EXQA-CBQA-TC-PARA"
DATA_DIR="${BASE_PATH}/data/"
CKPT_NAME="gpt2-large"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}"
SEED=${3-10}
SEED_ORDER=${4-10}
TYPE="many"
POS_TYPE=2
ICL_SUP=${5-test_target}
BATCH_SIZE=1
INNER_BATCH_SIZE=1024
EVAL_BATCH_SIZE=8
MAX_LENGTH=256
ICL_TRAIN_MAX_LENGTH=512
LR=0.00005
GRAD_ACC=16
SHOT=16

SAVE_PATH="${BASE_PATH}/icl_train/results_new/train/${DATA_NAMES}/${TYPE}_${POS_TYPE}/${ICL_SUP}/${SHOT}/${CKPT_NAME}/${SEED}-${SEED_ORDER}/lr${LR}_bs${BATCH_SIZE}_G${GRAD_ACC}_ml${MAX_LENGTH}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names ${DATA_NAMES}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --icl-inner-batch-size ${INNER_BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --train-iters 400"
OPTS+=" --save-interval 200"
OPTS+=" --eval-interval 200"
OPTS+=" --log-interval 1"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --icl-train-max-length ${ICL_TRAIN_MAX_LENGTH}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --lr ${LR}"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 2048"
OPTS+=" --epochs 10"
OPTS+=" --num-workers 4"
OPTS+=" --do-train"
# OPTS+=" --do-eval"
OPTS+=" --train-num 10000"
OPTS+=" --dev-num 100"
OPTS+=" --shot ${SHOT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
OPTS+=" --icl-sup ${ICL_SUP}"
OPTS+=" --unordered-pos-type ${POS_TYPE}"
OPTS+=" --type ${TYPE}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/icl_train/configs/deepspeed/ds_config.json"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/finetune_gpt2_many_ds.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
# ${CMD} 2>&1 | tee ${SAVE_PATH}/train.log
