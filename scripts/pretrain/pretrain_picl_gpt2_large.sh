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
CORPUS="100K_128"
DATA_PREFIX="picl/${CORPUS}_TRAIN_p1_en1_hn4_s42_lr5e-05-bs64-G1_4212.pt_L2_filtered_0.0"
LM_DATA_PREFIX="full_doc/"
DATA_DIR="${BASE_PATH}/pretrain_data/${CORPUS}/gpt2"
IDX_DATA_DIR="${BASE_PATH}/pretrain_data/${DATA_PREFIX}"
LM_DATA_DIR="${BASE_PATH}/pretrain_data/${LM_DATA_PREFIX}/gpt2"
# hp
BATCH_SIZE=1
LR=0.000001
LR_DECAY="noam"
GRAD_ACC=8
# length
MAX_LENGTH=1024
MAX_LENGTH_PER_SAMPLE=256
# runtime
SAVE_PATH="${BASE_PATH}/results/pretrain/"
# seed
SEED=10
SEED_ORDER=10
# icl
TYPE="vanilla"
ICL_SUP=all_target
EVAL_BATCH_SIZE=16
SHOT=16


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
OPTS+=" --picl-data-name picl"
OPTS+=" --picl-data-prefix ${DATA_PREFIX}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --lm-data-name lm"
OPTS+=" --lm-data-prefix ${LM_DATA_PREFIX}"
OPTS+=" --lm-ratio 1"
OPTS+=" --num-workers 4"
OPTS+=" --dev-ratio 0.5"
OPTS+=" --train-num 15000000"
OPTS+=" --pretrain-type mixed"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 1000"
OPTS+=" --lr-decay-style ${LR_DECAY}"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --loss-scale 2048"
OPTS+=" --epochs 1"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-length-per-sample ${MAX_LENGTH_PER_SAMPLE}"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
# OPTS+=" --do-eval"
OPTS+=" --save-interval 1000"
OPTS+=" --eval-interval 1000"
OPTS+=" --log-interval 1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# icl
OPTS+=" --shot ${SHOT}"
OPTS+=" --icl-sup ${ICL_SUP}"


export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/pretrain.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
