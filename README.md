## demo for BMKG

Requirements:

- torch ==1.11.0

#### to train

```shell
# /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/
# /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/
python ./main.py \
--data_root /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/ \
--use_cuda True \
--vocab_size 50265 \
--max_seq_len 512 \
--task_name triple \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--model_name bert_base \
--checkpoint_num 50 \
--batch_size 8 \
--learning_rate 1e-5 \
--gpu_ids 0 \
--soft_label False \
--weight_decay 0.0001 \
--use_pretrain False \
--save_path ./checkpoints/bert-base/ \
--epoch 10 \
--bmtrain False
```

```shell
# For BMTrain
MASTER_ADDR='103.242.175.227'
MASTER_PORT='22'
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1
NPROC_PER_NODE=4
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE ./main.py \
--data_root /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/ \
--use_cuda True \
--vocab_size 50265 \
--max_seq_len 512 \
--task_name triple \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--model_name bert_base \
--checkpoint_num 50 \
--batch_size 8 \
--learning_rate 1e-5 \
--gpu_ids 0 \
--soft_label False \
--weight_decay 0.0001 \
--use_pretrain False \
--save_path ./checkpoints/bert-base/ \
--epoch 10 \
--bmtrain True
```

