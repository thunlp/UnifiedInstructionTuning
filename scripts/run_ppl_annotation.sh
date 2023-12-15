#! /bin/bash

################################################################
# These are the only parameters you need to adjust for ppl annotation.
GPUS_PER_NODE=1
GPU="0"
MASTER_PORT=4300

SEED=42
BASE_PATH="/project/path"
CONFIG_PATH="/project/path/config"
CKPT_PATH="/project/path/config/pytorch_model.pt"
PY_PATH="/project/path/main.py"
# The parameters above should not be modified.
################################################################

DATASET="CB"
USE_DEMO=0
TRAIN_SET="none"
DEMO_FILE="none"
DEMO_PATH="${BASE_PATH}/data/CB/${DEMO_FILE}"
VERSION="none"
AVOID_REP=0
EVAL_ONLY=0
EVAL_NUM=-1
OPT_SHUFFLE=0
TXT_OUT_PATH=${BASE_PATH}/res/${DATASET}/${VERSION}/output.txt

BACKBONE_LR=0.0
PLUGIN_LR=0.00
TRAIN_EPOCH=8
TOTOL_BATCH_SIZE=8
EVAL_EPOCH_GAP=1
SAVE_EPOCH=-1
PLUG_PATH="none.pt"

WARM_UP=0
LR=0
DECODE_LENGTH=70
MASTER_ADDR=localhost
NNODES=1
NODE_RANK=0
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

OPTS=""
OPTS+=" --gpu ${GPU}"
OPTS+=" --eval-epoch-gap ${EVAL_EPOCH_GAP}"
OPTS+=" --ckpt-path ${CKPT_PATH}"
OPTS+=" --output-path ${TXT_OUT_PATH}"
OPTS+=" --decode-length ${DECODE_LENGTH}"

if [ ${EVAL_ONLY} != 0 ]
then
   OPTS+=" --eval-only"
fi
if [ ${AVOID_REP} != 0 ]
then
   OPTS+=" --avoid-rep"
fi
if [ ${USE_DEMO} != 0 ]
then
   OPTS+=" --demo"
fi
if [ ${EVAL_NUM} != -1 ]
then
   OPTS+=" --eval-num ${EVAL_NUM}"
fi
if [ ${OPT_SHUFFLE} != 0 ]
then
   OPTS+=" --option-shuffle"
fi

OPTS+=" --save-epoch ${SAVE_EPOCH}"
OPTS+=" --demo-path ${DEMO_PATH}"
OPTS+=" --dataset ${DATASET}"
OPTS+=" --train-set ${TRAIN_SET}"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --plug-path ${PLUG_PATH}"
OPTS+=" --save-name finetune-gptj-ckpt"
OPTS+=" --save ${BASE_PATH}/res/${DATASET}/${VERSION}"
OPTS+=" --model-config ${CONFIG_PATH}"

OPTS+=" --max-decoder-length 2048"
OPTS+=" --max-length 2048"
OPTS+=" --seed ${SEED}"
OPTS+=" --epochs ${TRAIN_EPOCH}"
OPTS+=" --lr ${LR}"  # TODO
OPTS+=" --backbone-lr ${BACKBONE_LR}"
# shellcheck disable=SC2003
OPTS+=" --batch-size $(expr $TOTOL_BATCH_SIZE / $GPUS_PER_NODE)"

OPTS+=" --save-iters 1000"
OPTS+=" --train-iters 2000"
OPTS+=" --inspect-iters 100"
OPTS+=" --warmup-iters ${WARM_UP}"
OPTS+=" --lr-decay-style constant"
OPTS+=" --plugin-lr ${PLUGIN_LR}"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 10.0"
OPTS+=" --loss-scale 1024"


CMD="torchrun ${DISTRIBUTED_ARGS} ${PY_PATH} ${OPTS}"
echo "${CMD}"
${CMD}