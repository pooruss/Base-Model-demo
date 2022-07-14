#!/bin/bash

frame=${1}
TASK=${2}
GPU_IDS=${3}
SAVE_PATH=${4}

export MAX_SEQ_LEN=192
export EPOCH=50
export BATCH_SIZE=64
export LR=0.00005
export NPROC_PER_NODE=4
export WEIGHT_DECAY=0.0001
export CKPT=20
export PRETRAINED_PATH=/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/
export DATA_ROOT=/data/private/wanghuadong/liangshihao/BMKG/data/WN18RR/


if [ $frame == 'bmtrain']
then
  MASTER_ADDR='103.242.175.227'
  MASTER_PORT='22'
  NNODES=1
  NODE_RANK=0
  GPUS_PER_NODE=1
  NPROC_PER_NODE=4
  torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE ./main.py \
  --data_root ${DATA_ROOT} \
  --use_cuda True \
  --max_seq_len ${MAX_SEQ_LEN} \
  --model_name bert_base \
  --checkpoint_num ${CKPT} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --gpu_ids ${GPU_IDS} \
  --weight_decay ${WEIGHT_DECAY} \
  --save_path ${SAVE_PATH} \
  --epoch ${EPOCH} \
  --bmtrain True \
  --task_name ${TASK} \
  --pretrained_path ${PRETRAINED_PATH}

else
  python ./main.py \
  --data_root ${DATA_ROOT} \
  --use_cuda True \
  --max_seq_len ${MAX_SEQ_LEN} \
  --model_name bert_base \
  --checkpoint_num ${CKPT} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --gpu_ids ${GPU_IDS} \
  --weight_decay ${WEIGHT_DECAY} \
  --save_path ${SAVE_PATH} \
  --epoch ${EPOCH} \
  --bmtrain False \
  --task_name ${TASK} \
  --pretrained_path ${PRETRAINED_PATH}

fi




