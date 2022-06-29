## demo for BMKG

Requirements:

- torch ==1.11.0

#### to train

```shell
python ./main.py \
--data_root /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/ \
--use_cuda True \
--vocab_size 50265 \
--max_seq_len 1024 \
--task_name triple \
--pretrained_embed_path ./checkpoints/transe.ckpt \
--use_ema False \
--model_name coke \
--checkpoint_num 50 \
--batch_size 512 \
--learning_rate 1e-3 \
--gpu_ids 0 \
--soft_label False \
--weight_decay 0.0001 \
--use_pretrain False \
--save_path ./checkpoints/ \
--epoch 200 \
--bmtrain True
```

