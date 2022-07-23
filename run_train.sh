#!/bin/bash

frame=${1}
TASK=${2}
DATA_DIRE=${3}
GPU_IDS=${4}
SAVE_PATH=${5}

export MAX_SEQ_LEN=187
export EPOCH=10
export BATCH_SIZE=20
export LR=0.00005
export WEIGHT_DECAY=0.0001
export CKPT=20
export PRETRAINED_PATH=/home/wanghuadong/liangshihao/KEPLER-huggingface/roberta-base/
export DATA_ROOT=/data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237-demo/
export CKPT_PATH=./checkpoints/bert-basebert_base_lr5e-05_bs128_emaFalse_epoch3.pt
export USE_CKPT=False
export USE_ROBERTA=True
export MODEL_NAME=roberta_base

if [ $frame = 'bmtrain' ]
then
  MASTER_ADDR='103.242.175.227'
  MASTER_PORT='22'
  NNODES=1
  NODE_RANK=1
  GPUS_PER_NODE=1
  NPROC_PER_NODE=8
  torchrun --standalone --nnodes=1 --rdzv_id=10 --nproc_per_node=$NPROC_PER_NODE ./main.py \
  --data_root ${DATA_ROOT} \
  --use_cuda True \
  --max_seq_len ${MAX_SEQ_LEN} \
  --model_name ${MODEL_NAME} \
  --checkpoint_num ${CKPT} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --gpu_ids ${GPU_IDS} \
  --weight_decay ${WEIGHT_DECAY} \
  --save_path ${SAVE_PATH} \
  --epoch ${EPOCH} \
  --bmtrain True \
  --task_name ${TASK} \
  --data_directory ${DATA_DIRE} \
  --pretrained_path ${PRETRAINED_PATH} \
  --do_train True \
  --do_test False \
  --ckpt_path ${CKPT_PATH} \
  --use_ckpt ${USE_CKPT} \
  --roberta ${USE_ROBERTA}

else
  python ./main.py \
  --data_root ${DATA_ROOT} \
  --use_cuda True \
  --max_seq_len ${MAX_SEQ_LEN} \
  --model_name ${MODEL_NAME} \
  --checkpoint_num ${CKPT} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --gpu_ids ${GPU_IDS} \
  --weight_decay ${WEIGHT_DECAY} \
  --save_path ${SAVE_PATH} \
  --epoch ${EPOCH} \
  --bmtrain False \
  --task_name ${TASK} \
  --data_directory ${DATA_DIRE} \
  --pretrained_path ${PRETRAINED_PATH} \
  --do_train True \
  --do_test False \
  --ckpt_path ${CKPT_PATH} \
  --use_ckpt ${USE_CKPT} \
  --roberta ${USE_ROBERTA}
fi




