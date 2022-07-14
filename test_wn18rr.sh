#!/bin/bash

frame=${1}
TASK=${2}
GPU_IDS=${3}
ckpt_path=${4}
save_path=${5}

export MAX_SEQ_LEN=192
export BATCH_SIZE=64
export NPROC_PER_NODE=4
export DATA_ROOT=/data/private/wanghuadong/liangshihao/BMKG/data/WN18RR/
export PRETRAINED_PATH=/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/
export MAX_ANS_LEN=17


if [ ${frame} = 'bmtrain'];
then
  MASTER_ADDR='103.242.175.227'
  MASTER_PORT='22'
  NNODES=1
  NODE_RANK=0
  GPUS_PER_NODE=1
  NPROC_PER_NODE=4
  torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE ./evaluation.py \
  --data_root ${DATA_ROOT} \
  --batch_size ${BATCH_SIZE} \
  --use_cuda True \
  --max_seq_len ${MAX_SEQ_LEN} \
  --max_ans_len ${MAX_ANS_LEN} \
  --model_name bert_base \
  --gpu_ids ${GPU_IDS} \
  --bmtrain False \
  --task_name ${TASK} \
  --ckpt_path ${ckpt_path} \
  --save_path ${save_path} \
  --pretrained_path ${PRETRAINED_PATH}

else
  python ./evaluation.py \
  --data_root ${DATA_ROOT} \
  --batch_size ${BATCH_SIZE} \
  --use_cuda True \
  --max_seq_len ${MAX_SEQ_LEN} \
  --max_ans_len ${MAX_ANS_LEN} \
  --model_name bert_base \
  --gpu_ids ${GPU_IDS} \
  --bmtrain False \
  --task_name ${TASK} \
  --ckpt_path ${ckpt_path} \
  --save_path ${save_path} \
  --pretrained_path ${PRETRAINED_PATH}
fi



