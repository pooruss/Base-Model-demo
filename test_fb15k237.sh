#!/bin/bash

frame=${1}
TASK=${2}
DATA_DIRE=${3}
GPU_IDS=${4}
ckpt_path=${5}
save_path="./checkpoints/"
save_file="1.json"

export MAX_SEQ_LEN=187
export BATCH_SIZE=4
export DATA_ROOT=/data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237-demo/
export PRETRAINED_PATH=/home/wanghuadong/liangshihao/KEPLER-huggingface/roberta-base/
export MAX_ANS_LEN=21
export ROBERTA=True

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
  --data_directory ${DATA_DIRE} \
  --ckpt_path ${ckpt_path} \
  --save_path ${save_path} \
  --save_file ${save_file} \
  --pretrained_path ${PRETRAINED_PATH} \
  --do_train False \
  --do_test True \
  --roberta ${ROBERTA}

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
  --data_directory ${DATA_DIRE} \
  --ckpt_path ${ckpt_path} \
  --save_path ${save_path} \
  --save_file ${save_file} \
  --pretrained_path ${PRETRAINED_PATH} \
  --do_train False \
  --do_test True \
  --roberta ${ROBERTA}
fi
