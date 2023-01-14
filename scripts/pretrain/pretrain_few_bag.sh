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

BASE_PATH=${1-"/home/guyuxian/CodeRepo"}
DATA_DIR="${BASE_PATH}/icl_train/unsup_data/general/para_corpus_10M_uniq_shuf/256/ret_split_w_label/bs64_lr0.00005_G1_SEED10/5411.pt/l2_h5/full/res_-1_filtered/"
CKPT_NAME="gpt2-large"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
SEED=${3-10}
SEED_ORDER=${4-10}
TYPE="few_unordered"
POS_TYPE=2
ICL_SUP=${5-test_target}
BATCH_SIZE=2
EVAL_BATCH_SIZE=32
MAX_LENGTH=1024
LR=0.00005
LR_DECAY="noam"
GRAD_ACC=128
SHOT=16

SAVE_PATH="${BASE_PATH}/icl_train/results_new/pretrain/${DATA_NAMES}/${TYPE}/${ICL_SUP}/${SHOT}/${CKPT_NAME}/${SEED}-${SEED_ORDER}/lr${LR}_${LR_DECAY}_fix_bs${BATCH_SIZE}_G${GRAD_ACC}_ml${MAX_LENGTH}"

OPTS=""
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-config ${CKPT}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --unsup-data-name tokenized"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --train-iters 400"
OPTS+=" --save-interval 1000"
OPTS+=" --eval-interval 1000"
OPTS+=" --log-interval 1"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --lr ${LR}"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style ${LR_DECAY}"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 2048"
OPTS+=" --epochs 1"
OPTS+=" --num-workers 4"
OPTS+=" --do-train"
# OPTS+=" --do-eval"
OPTS+=" --shot ${SHOT}"
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
OPTS+=" --icl-sup ${ICL_SUP}"
OPTS+=" --unordered"
OPTS+=" --unordered-pos-type ${POS_TYPE}"
OPTS+=" --type ${TYPE}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/icl_train/configs/deepspeed/ds_config.json"
# OPTS+=" --gradient-checkpointing"

export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/icl_train/pretrain_gpt2_few_ds.py ${OPTS}"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
# ${CMD} 2>&1 | tee ${SAVE_PATH}/train.log
