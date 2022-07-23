import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.bigmodel import BertBase, BertBaseBMT, RobertaBaseBMT, RobertaBase
from data.base_dataset import BaseDataset, BaseDevDataset, NegativeDataset
from config import init_model_config, init_train_config
from trainer import Trainer
from trainer_alias import Trainer_alias
import math
import logging
import argparse
from config.args import ArgumentGroup
from model_center.dataset import DistributedDataLoader
from data import MLERMFeatures

# import wandb

# wandb.init(project="CoKE", entity="pooruss")


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
# model_g.add_arg("model_name", str, "coke_roberta", "Model name")

model_g.add_arg("initializer_range", int, 0.02, "CoKE model config: initializer_range")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("node", int, 1, "Node nums.")
train_g.add_arg("neg_nums", int, 3, "Negatives nums.")
train_g.add_arg("warmup_epoch", int, 1, "Number of epoches for training.")
train_g.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
# train_g.add_arg("weight_decay", float, 0.0001, "Weight decay rate for L2 regularizer.")
train_g.add_arg("lr_scheduler", str, "step_lr", "warmup_cosine, linear_lr")
train_g.add_arg("use_ema", bool, False, "Use ema.")
train_g.add_arg("ema_decay", float, 0.999, "Ema decay.")

parser.add_argument("--task_name", default='path', type=str, required=True, help="path or triple.")
parser.add_argument("--data_directory", default='path', type=str, required=True, help="path or triple.")
parser.add_argument("--data_root", default='/home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/', type=str,
                    required=True, help="data directory.")
parser.add_argument("--max_seq_len", default=7, type=int, required=True, help="sequence length.")
parser.add_argument("--epoch", default=400, type=int, required=True, help="epoch.")
parser.add_argument("--use_cuda", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--batch_size", default=2048, type=int, required=True, help="batch size.")
parser.add_argument("--learning_rate", default=2e-3, type=float, required=True, help="lr.")
parser.add_argument("--weight_decay", default=0.0001, type=float, required=True, help="lr.")
parser.add_argument("--checkpoint_num", default=50, type=int, required=True, help="ckpt nums")
parser.add_argument("--save_path", default='./checkpoints/', type=str, required=True, help="save directory.")
parser.add_argument("--pretrained_path", default='./', type=str, required=True, help="pretrained path.")
parser.add_argument("--gpu_ids", default='0', type=str, required=True, help="gpu ids.")
parser.add_argument("--model_name", default='coke', type=str, required=True, help="coke or coke_roberta.")
parser.add_argument("--bmtrain", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--do_train", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--do_test", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--mrm", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--ckpt_path", default='./', type=str, required=True, help="pretrained path.")
parser.add_argument("--use_ckpt", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--roberta", default=True, type=boolean_string, help="use bmtrain or not.")

args = parser.parse_args()

# wandb.config.update(args)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1,3,4,5'


def main():
    print(args.use_cuda)
    if args.bmtrain:
        import bmtrain as bmt
        bmt.init_distributed(seed=0)

    # ------------
    # data
    # ------------
    if args.neg_nums > 0:
        train_dataset = NegativeDataset(data_dir=args.data_root,
                            do_train=True,
                            do_eval=False,
                            do_test=False,
                            task=args.task_name,
                            data_directory=args.data_directory,
                            max_seq_len=args.max_seq_len,
                            pretrained_path=args.pretrained_path)
    else:
        train_dataset = BaseDataset(data_dir=args.data_root,
                                do_train=True,
                                do_eval=False,
                                do_test=False,
                                task=args.task_name,
                                data_directory=args.data_directory,
                                max_seq_len=args.max_seq_len,
                                pretrained_path=args.pretrained_path)
    val_dataset = BaseDevDataset(data_dir=args.data_root,
                            do_train=False,
                            do_eval=True,
                            do_test=False,
                            task=args.task_name,
                            data_directory=args.data_directory,
                            max_seq_len=args.max_seq_len,
                            pretrained_path=args.pretrained_path)

    if args.bmtrain:
        train_loader = DistributedDataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DistributedDataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    args.padding_id = train_dataset.pad_id

    # ------------
    # model
    # ------------
    device_ids = list()
    for gpu_id in args.gpu_ids:
        device_ids.append(int(gpu_id))
    device = torch.device('cuda:{}'.format(device_ids[0]) if args.use_cuda else 'cpu')
    args.gpus = len(device_ids)

    args.mask_id = train_dataset.mask_id
    args.vocab_size = train_dataset.vocab_size

    model_config = init_model_config(args, logger, print_config=True)
    if args.bmtrain:
        if args.roberta:
            model = RobertaBaseBMT(config=model_config)
        else:
            model = BertBaseBMT(config=model_config)
        bmt.init_parameters(model)
        bmt.synchronize()
    else:
        if args.roberta:
            model = RobertaBase(config=model_config)
        else:
            model = BertBase(config=model_config)
        model.tokenizer = train_dataset.tokenizer
        if args.gpus > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        # state_dict = torch.load(args.ckpt_path, map_location=device)
        # for name, param in state_dict.items():
        #     if "linear2" in name:
        #         print(param)
        # model.load_state_dict(state_dict)
        model.to(device=device)
    # ------------
    # training
    # ------------
    train_config = init_train_config(args, logger, print_config=True)
    if args.bmtrain:
        param_group = [{'params': model.roberta.parameters(), 'lr':args.learning_rate},
                       {'params': model.lm_head.parameters(), 'lr':args.learning_rate * 10}]
        optimizer = bmt.optim.AdamOptimizer(param_group, lr=args.learning_rate,
                                            weight_decay=args.weight_decay)
        scheduler = bmt.lr_scheduler.Noam(
            optimizer,
            start_lr=args.learning_rate * 5,
            warmup_iter=args.warmup_epoch,
            end_iter=args.epoch,
            num_iter=args.warmup_epoch / 2
        )
        loss_function = bmt.loss.FusedCrossEntropy(ignore_index=train_dataset.pad_id)
        bmt.synchronize()
    else:
        param_group = [{'params': model.model.parameters(), 'lr':args.learning_rate},
                       {'params': model.lm_head.parameters(), 'lr':args.learning_rate * 10}]
        optimizer = optim.Adam(param_group, weight_decay=args.weight_decay)
        # warm_up_with_cosine_lr
        t = args.warmup_epoch  # warmup
        T = args.epoch
        n_t = 0.5
        lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))
        if args.lr_scheduler == "warmup_cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        elif args.lr_scheduler == "linear_lr":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epoch)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        loss_function = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id, label_smoothing=0.0)
    # for name, parms in model.named_parameters():
    #     print('-->name:', name)

    if 'alias' in args.task_name:
        model_trainer = Trainer_alias(model, train_config)
    else:
        model_trainer = Trainer(model, train_config)

    model_trainer.load_train_data_loader(train_loader)
    model_trainer.load_val_data_loader(val_loader)

    model_trainer.set_loss_function(loss_function)
    model_trainer.set_optimizer(optimizer)
    model_trainer.set_lr_scheduler(scheduler)
    # model_trainer.set_pretrained_lr_scheduler(pretrained_scheduler)

    model_trainer.train()


if __name__ == '__main__':
    main()
