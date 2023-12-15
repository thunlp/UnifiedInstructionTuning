#! /bin/bash

#######################################################
# Following are the parameters to adjust.
GPUS_PER_NODE=1
GPU="0"
MASTER_PORT=$(shuf -i25000-30000 -n1)
MASTER_ADDR=localhost

NNODES=1
NODE_RANK=0
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

TRAIN_EPOCH=2
TOTAL_BATCH_SIZE=16
EVAL_EPOCH_GAP=100
SAVE_EPOCH=100

BASE_PATH="/project/path"
CKPT_PATH="project/path/config/pytorch_model.pt"
CONFIG_PATH="project/path/config"

SEED=502
DATASET="template"
VERSION="ddmm_name"
TXT_OUT_PATH=${BASE_PATH}/res/${VERSION}/output.txt

WARM_UP=0
BACKBONE_LR=1e-6
LR=1e-6
DECODE_LENGTH=328
#######################################################

OPTS=""
OPTS+=" --gpu ${GPU}"
OPTS+=" --eval-epoch-gap ${EVAL_EPOCH_GAP}"
OPTS+=" --ckpt-path ${CKPT_PATH}"
OPTS+=" --output-path ${TXT_OUT_PATH}"
OPTS+=" --decode-length ${DECODE_LENGTH}"

OPTS+=" --save-epoch ${SAVE_EPOCH}"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --save-name finetune-gptj-ckpt"
OPTS+=" --save ${BASE_PATH}/res/${VERSION}"
OPTS+=" --model-config ${CONFIG_PATH}"

OPTS+=" --max-decoder-length 1024"
OPTS+=" --seed ${SEED}"
OPTS+=" --epochs ${TRAIN_EPOCH}"
OPTS+=" --lr ${LR}"
OPTS+=" --backbone-lr ${BACKBONE_LR}"
# shellcheck disable=SC2003
OPTS+=" --batch-size $(expr $TOTAL_BATCH_SIZE / $GPUS_PER_NODE)"

OPTS+=" --save-iters 1000"
OPTS+=" --train-iters 2000"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters ${WARM_UP}"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 1024"

CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/main.py ${OPTS}"
echo "${CMD}"
mkdir "${BASE_PATH}"/res/${VERSION}/
${CMD} 2>&1 | tee "${BASE_PATH}/res/${VERSION}/train_log.txt"