import os
import json
import numpy as np
from tqdm import tqdm
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.bigmodel import CoKE, CoKE_Roberta, CoKE_BMT
from data.base_dataset import BaseDataset
from config import init_coke_net_config
import logging
import argparse
from config.args import ArgumentGroup
import bmtrain as bmt
from model_center.dataset import DistributedDataLoader


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

task_g = ArgumentGroup(parser, "task", "which task to run.")
task_g.add_arg("do_train", bool, True, "Train")
task_g.add_arg("do_val", bool, True, "Validation")
task_g.add_arg("do_test", bool, False, "Test")

model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
# model_g.add_arg("model_name", str, "coke_roberta", "Model name")
model_g.add_arg("hidden_size", int, 256, "CoKE model config: hidden size, default 256")
model_g.add_arg("num_hidden_layers", int, 12, "CoKE model config: num_hidden_layers, default 12")
model_g.add_arg("num_attention_heads", int, 4, "CoKE model config: num_attention_heads, default 4")
model_g.add_arg("num_relations", int, 1345, "CoKE model config: vocab_size")
model_g.add_arg("max_position_embeddings", int, 1024, "CoKE model config: max_position_embeddings")
model_g.add_arg("dropout", float, 0.1, "CoKE model config: dropout, default 0.1")
model_g.add_arg("hidden_dropout", float, 0.1, "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("attention_dropout", float, 0.1,
                "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("initializer_range", int, 0.02, "CoKE model config: initializer_range")
model_g.add_arg("intermediate_size", int, 512, "CoKE model config: intermediate_size, default 512")
model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("node", int, 1, "Node nums.")
train_g.add_arg("warmup_epoch", int, 4, "Number of epoches for training.")
train_g.add_arg("warmup_proportion", float, 0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
# train_g.add_arg("weight_decay", float, 0.0001, "Weight decay rate for L2 regularizer.")
train_g.add_arg("ema_decay", float, 0.9999, "EMA decay.")
train_g.add_arg("lr_scheduler", str, "warmup_cosine", "linear_lr, step_lr")

log_g = ArgumentGroup(parser, "logging", "logging related.")
log_g.add_arg("skip_steps", int, 30, "The steps interval to print loss.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("dataset", str, "FB15K", "dataset name")
data_g.add_arg("vocab_file", str, 'vocab.txt', "Vocab path.")
data_g.add_arg("train_file", str, 'train.coke.txt', "Train data for coke.")
data_g.add_arg("valid_file", str, 'valid.coke.txt', "Valid data for coke.")
# data_g.add_arg("test_file", str, 'test.coke.txt', "Test data for coke.")
data_g.add_arg("dev_file", str, 'dev.coke.txt', "Dev data for coke.")
data_g.add_arg("true_triple_file", str, 'all.txt', "All triple data for coke.")
data_g.add_arg("sen_candli_file", str, 'sen_candli.txt',
               "sentence_candicate_list file for path query evaluation. Only used for path query datasets")
data_g.add_arg("sen_trivial_file", str, 'trival_sen.txt',
               "trivial sentence file for pathquery evaluation. Only used for path query datasets")

parser.add_argument("--task_name", default='path', type=str, required=True, help="path or triple.")
parser.add_argument("--data_root", default='/home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/', type=str,
                    required=True, help="data directory.")
parser.add_argument("--vocab_size", default=16396, type=int, required=True, help="16396 for fb15k, 75169 for pathFB.")
parser.add_argument("--max_seq_len", default=7, type=int, required=True, help="sequence length.")
parser.add_argument("--epoch", default=400, type=int, required=True, help="epoch.")
parser.add_argument("--use_cuda", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--batch_size", default=2048, type=int, required=True, help="batch size.")
parser.add_argument("--learning_rate", default=2e-3, type=float, required=True, help="lr.")
parser.add_argument("--weight_decay", default=0.0001, type=float, required=True, help="lr.")
parser.add_argument("--checkpoint_num", default=50, type=int, required=True, help="ckpt nums")
parser.add_argument("--save_path", default='./checkpoints/', type=str, required=True, help="save directory.")
parser.add_argument("--gpu_ids", default='0', type=str, required=True, help="gpu ids.")
parser.add_argument("--model_name", default='coke', type=str, required=True, help="coke or coke_roberta.")
parser.add_argument("--soft_label", default=False, type=boolean_string, help="soft label or not.")
parser.add_argument("--pretrained_embed_path", default='./checkpoints/transe.ckpt', type=str, help="pretrained embed.")
parser.add_argument("--use_ema", default=False, type=boolean_string, help="use ema or not.")
parser.add_argument("--bmtrain", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--use_pretrain", default=False, type=boolean_string, required=True, help="lr.")
parser.add_argument("--ckpt_root", default='.', type=str,
                    required=True, help="save directory.")

args = parser.parse_args()

# wandb.config.update(args)


def eval_fn(target, pred_top10):
    index_list = []
    for label, pred in zip(target, pred_top10):
        if label in pred:
            this_idx = np.where(pred == label)[0][0]
        else:
            this_idx = 10
        index_list.append(this_idx)
    index_list = np.array(index_list)
    # print('index_list: ', index_list)
    hits1 = float(len(index_list[index_list < 1])) / len(index_list)
    hits3 = float(len(index_list[index_list < 3])) / len(index_list)
    hits10 = float(len(index_list[index_list < 10])) / len(index_list)
    MR = np.mean(index_list) + 1
    MRR = np.mean([1 / (x + 1) for x in index_list])
    return hits1, hits3, hits10, MR, MRR


def main():
    if args.bmtrain:
        bmt.init_distributed(seed=0)
    # ------------
    # data
    # ------------
    test_dataset = BaseDataset(data_dir=args.data_root,
                               do_train=False,
                               do_eval=False,
                               do_test=True,
                               max_seq_len=512)
    if args.bmtrain:
        test_data_loader = DistributedDataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ------------
    # model
    # ------------
    device_ids = list()
    for gpu_id in args.gpu_ids:
        device_ids.append(int(gpu_id))
    device = torch.device('cuda:{}'.format(device_ids[0]) if args.use_cuda else 'cpu')
    args.gpus = len(device_ids)

    args.mask_id = test_dataset.mask_id
    args.vocab_size = test_dataset.vocab_size

    coke_config = init_coke_net_config(args, logger, print_config=True)
    if args.model_name == 'coke_roberta':
        model = CoKE_Roberta(config=coke_config)
    elif args.bmtrain:
        model = CoKE_BMT(config=coke_config)
        # if args.use_pretrain:
        #     model.load_pretrained_embedding(args.pretrained_embed_path)
        bmt.init_parameters(model)
        # bmt.load(model, args.ckpt_root)
        bmt.synchronize()
    else:
        model = CoKE(config=coke_config)
        model.tokenizer = test_dataset.tokenizer
        if args.use_pretrain:
            model.load_pretrained_embedding(args.pretrained_embed_path)

        if args.gpus > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device=device)

    # ------------
    # evaluation
    # ------------
    # model.eval()
    print("----------- evaluation -----------")

    ent2alias = {}
    with open(os.path.join(args.data_root, "entity2text.txt"), 'r', encoding='utf-8') as f:
        ent_lines = f.readlines()
        for line in ent_lines:
            temp = line.strip().split('\t')
            if len(temp) == 2:
                end = temp[1]  # .find(',')
                ent2alias[temp[0]] = temp[1]  # [:end]

    ent2text = {}
    id2entity = {}
    entity2token_id = {}
    if args.data_root.find("FB15") != -1:
        with open(os.path.join(args.data_root, "entity2textlong.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for idx, line in enumerate(ent_lines):
                temp = line.strip().split('\t')
                # first_sent_end_position = temp[1].find(".")
                ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]
                id2entity[idx] = temp[0]
                entity = test_dataset.tokenizer.tokenize(ent2alias[temp[0]])
                entity2token_id[temp[0]] = test_dataset.tokenizer.convert_tokens_to_ids(entity)
    # entities = list(ent2text.keys())

    topk_entity_list = []
    test_lines = test_dataset.get_test_triples(args.data_root)
    test = []
    for line in test_lines:
        head, rel, tail = line
        test.append(tail)

    for iter, batch_data in tqdm(enumerate(test_data_loader)):
        # fetch batch data
        targets = []
        try:
            src_ids, input_mask, seg_ids, pos_ids, mlm_label_ids, \
            mem_label_ids,  mem_label_idx = batch_data
            targets = mem_label_ids
        except RuntimeError:
            print("One data instance's length should be 5, received {}.".format(len(batch_data)))
            continue
        if args.use_cuda:
            src_ids, input_mask, seg_ids, pos_ids, mlm_label_ids, mem_label_ids, mem_label_idx = \
                src_ids.to(device), input_mask.to(device), seg_ids.to(device), \
                pos_ids.to(device), mlm_label_ids.to(device), mem_label_ids.to(device), mem_label_idx.to(device)

        with torch.no_grad():
            mlm_mask_pos = torch.argwhere(mlm_label_ids.reshape(-1) > 0)
            mem_mask_pos = torch.argwhere(mem_label_ids.reshape(-1) > 0)
            mem_label_ids = mem_label_ids.view(-1)[mem_mask_pos].squeeze()
        input_x = {
            'src_ids': src_ids,
            'input_mask': input_mask,
            'segment_ids': seg_ids,
            'position_ids': pos_ids,
            'mlm_mask_pos': mlm_mask_pos,
            'mem_mask_pos': mem_mask_pos
        }
        torch.cuda.synchronize()

        # forward
        mem_probs = model(input_x)["logits"]
        mem_probs = mem_probs.cpu().detach().numpy()
        mem_label_idx = mem_label_idx.cpu().detach().numpy()
        for j in range(len(mem_probs)):
            # print('output_prob: ', output_prob.shape)
            preds = mem_probs[j][mem_label_idx[j][0]+1: mem_label_idx[j][1]+1]
            target = targets[j][mem_label_idx[j][0]+1: mem_label_idx[j][1]+1]
            # target = np.swapaxes(target, 0, 1)
            # preds = np.argmax(preds, axis=1)
            entity_prob_list = []
            for _entity_id in range(len(id2entity)):
                _entity_token_id = entity2token_id[id2entity[_entity_id]]

                _entity_prob = preds[np.arange(len(_entity_token_id)), _entity_token_id]
                # print('_entity_prob: ', _entity_prob)
                entity_prob_list.append(np.mean(_entity_prob))
            topk_entity_idx = np.argsort(entity_prob_list)[::-1]  # [:10]
            topk_entity = [id2entity[_id] for _id in topk_entity_idx]
            topk_entity_list.append(topk_entity)
            topk_entity_array = np.array(topk_entity_list)
            test_hits1, test_hits3, test_hits10, test_MR, test_MRR = eval_fn(test, topk_entity_array)
            print(test_hits10)
            print(test_MRR)
    topk_entity_array = np.array(topk_entity_list)
    test_hits1, test_hits3, test_hits10, test_MR, test_MRR = eval_fn(test, topk_entity_array)
    print(test_MRR)
    # eval_result_file = os.path.join(args.save_path, "eval_result.json")
    # outs = "%.3f\t%.3f\t%.3f\t%.3f" % (
    #     eval_result['fmrr'], eval_result['fhits1'], eval_result['fhits3'], eval_result['fhits10'])
    # logger.info("\n----------- Evaluation Performance --------------\n%s\n%s" %
    #             ("\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@10"]), outs))


if __name__ == '__main__':
    main()
