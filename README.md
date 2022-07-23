## demo for BMKG

Requirements:

- torch ==1.11.0



#### data preprocess

```shell
# /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/
# /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/
sh data_preprocess.sh fb15k237-roberta triple_text /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237-demo/ triple_text/
sh data_preprocess.sh fb15k237 triple_text /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237-demo/ triple_text/
sh data_preprocess.sh wn18rr triple_text /data/private/wanghuadong/liangshihao/BMKG/data/WN18RR/ triple_text/
```

#### to train
```shell
# /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/
# /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/
sh train_fb15k237.sh torch triple_text triple_text/ 0 ./checkpoints/
sh train_wnn18rr.sh torch triple_text triple_text/ 0 ./checkpoints/
sh train_fb15k237.sh bmtrain triple_text triple_text/ 0 ./checkpoints/
```


#### to test

```shell
# /data/private/wanghuadong/liangshihao/BMKG/data/FB15k-237/
# /home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/
sh test_fb15k237.sh torch triple_text triple_text/ 0 ckpt_dir
sh test_wn18rr.sh torch triple_text triple_text/ 0 ckpt_dir
```

