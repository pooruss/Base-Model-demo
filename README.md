## demo for BMKG

Requirements:

- torch ==1.11.0
- transformers==4.4.0
- bmtrain==0.1.7



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
sh run_train.sh torch triple_text triple_text/ 0 ./checkpoints/
# use bmtrain
sh run_train.sh bmtrain triple_text triple_text/ 0 ./checkpoints/
```


#### to test

```shell
sh run_test.sh torch triple_text triple_text/ 0 ckpt_dir
```

