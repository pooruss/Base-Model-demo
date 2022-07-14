## demo for BMKG

Requirements:

- torch ==1.11.0



#### data preprocess

```shell
# /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/
# /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/
sh data_preprocess.sh fb15k237 triple_text /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237-demo/ triple_text/
sh data_preprocess.sh wn18rr triple_text /data/private/wanghuadong/liangshihao/BMKG/data/WN18RR/ triple_text/
```

#### to train
```shell
# /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/
# /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/
sh train_fb15k237.sh torch triple_text 0 /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237-demo/
sh train_wn18rr.sh torch triple_text 0 /data/private/wanghuadong/liangshihao/BMKG/data/WN18RR/
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
--data_root /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237-demo/ \
--use_cuda True \
--vocab_size 50265 \
--max_seq_len 512 \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--model_name bert_base \
--checkpoint_num 50 \
--batch_size 128 \
--learning_rate 1e-3 \
--gpu_ids 0 \
--soft_label False \
--weight_decay 0.0001 \
--use_pretrain False \
--save_path ./checkpoints/bert-base/ \
--epoch 50 \
--bmtrain True \
--task_name path_text
```

#### to test

```shell
# /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/
# /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/
sh test_fb15k237.sh torch triple_text 0 ckpt_dir result_save_dir result_file_name
sh test_wn18rr.sh torch triple_text 0 ckpt_dir result_save_dir result_file_name
```

