import sys
import os
import json
import numpy as np
from tqdm import tqdm
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.bigmodel import BertBase, BertBaseBMT
from data.base_dataset import BaseDataset, BaseTestDataset
from config import init_bert_net_config
import logging
import argparse
from config.args import ArgumentGroup
import bmtrain as bmt
from model_center.dataset import DistributedDataLoader
from data import MLMFeaturesWithNeg, MLERMFeatures


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

parser.add_argument("--data_root", default='/home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/', type=str,
                    required=True, help="data directory.")
parser.add_argument("--max_seq_len", default=7, type=int, required=True, help="sequence length.")
parser.add_argument("--max_ans_len", default=15, type=int, required=True, help="sequence length.")
parser.add_argument("--use_cuda", default=True, type=boolean_string, required=True, help="gpu or cpu.")
parser.add_argument("--batch_size", default=2048, type=int, required=True, help="batch size.")
parser.add_argument("--task_name", default='path_alias', type=str, required=True, help="save directory.")
parser.add_argument("--gpu_ids", default='0', type=str, required=True, help="gpu ids.")
parser.add_argument("--model_name", default='coke', type=str, required=True, help="coke or coke_roberta.")
parser.add_argument("--bmtrain", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--ckpt_path", default='.', type=str, required=True, help="save directory.")
parser.add_argument("--save_path", default='.', type=str, required=True, help="save directory.")
parser.add_argument("--save_file", default='.', type=str, required=True, help="save directory.")
parser.add_argument("--pretrained_path", default='.', type=str, required=True, help="save directory.")
parser.add_argument("--do_train", default=False, type=boolean_string, help="use bmtrain or not.")
parser.add_argument("--do_test", default=False, type=boolean_string, help="use bmtrain or not.")

args = parser.parse_args()


# wandb.config.update(args)


def eval_fn(target, pred_top10, mr, mrr):
    index_list = []
    addition = 0
    for label, pred in zip(target, pred_top10):
        if label in pred:
            this_idx = np.where(pred == label)[0][0]
        else:
            addition += 1
            continue
        index_list.append(this_idx)
    index_list = np.array(index_list)
    # print('index_list: ', index_list)
    hits1 = float(len(index_list[index_list < 1])) / (len(index_list) + addition)
    hits3 = float(len(index_list[index_list < 3])) / (len(index_list) + addition)
    hits10 = float(len(index_list[index_list < 10])) / (len(index_list) + addition)
    if len(index_list) != 0:
        MR = np.mean(index_list) + 1
        MRR = np.mean([1 / (x + 1) for x in index_list])
    else:
        MR = mr
        MRR = mrr
    return hits1, hits3, hits10, MR, MRR


def main():
    if args.bmtrain:
        bmt.init_distributed(seed=0)
    # ------------
    # data
    # ------------
    test_dataset = BaseTestDataset(data_dir=args.data_root,
                                   do_train=False,
                                   do_eval=False,
                                   do_test=True,
                                   task=args.task_name,
                                   max_seq_len=args.max_seq_len)
    if args.bmtrain:
        test_data_loader = DistributedDataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # build test
    ent2alias = {}
    ent2text = {}
    entity2token_id = {}
    mean_entity_len = 0.0
    with open(os.path.join(args.data_root, "entity2text.txt"), 'r', encoding='utf-8') as f:
        total_len = 0.0
        ent_lines = f.readlines()
        for line in ent_lines:
            temp = line.strip().split('\t')
            if len(temp) == 2:
                if args.data_root.find("FB15") != -1:
                    ent2alias[temp[0]] = temp[1]
                elif args.data_root.find("WN18") != -1:
                    end = temp[1].find(',')
                    ent2alias[temp[0]] = temp[1][:end]
                    ent2text[temp[0]] = temp[1][end:]
                entity = test_dataset.tokenizer.tokenize(ent2alias[temp[0]])
                entity2token_id[temp[0]] = test_dataset.tokenizer.convert_tokens_to_ids(entity)
                entity_len = len(entity2token_id[temp[0]])
                total_len += entity_len
        print(total_len/len(ent_lines))

    if args.data_root.find("FB15") != -1:
        ent2text = {}
        with open(os.path.join(args.data_root, "entity2textlong.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                # first_sent_end_position = temp[1].find(".")
                ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]
    id2entity = {}
    entities = list(ent2text.keys())
    for idx, entity in enumerate(entities):
        id2entity[idx] = entity

    all_lines = test_dataset.get_all_triples(args.data_root)
    filter_dict = {}
    for line in all_lines:
        if len(line) == 3:
            head, rel, tail = line
        else:
            head, rel, tail = line[0], line[1] + line[3], line[4]
        if head not in filter_dict:
            filter_dict[head] = {}
        if rel not in filter_dict[head]:
            filter_dict[head][rel] = []
        filter_dict[head][rel].append(tail)

        if tail not in filter_dict:
            filter_dict[tail] = {}
        if rel not in filter_dict[tail]:
            filter_dict[tail][rel] = []
        filter_dict[tail][rel].append(head)

    test_lines = test_dataset.get_test_triples(args.data_root)
    test = []
    test_rel = []
    mask_type_list = []
    mean_entity_len = 0.0
    total_len = 0
    for line in test_lines:
        if len(line) == 3:
            head, rel, tail = line
        else:
            head, rel, tail = line[0], line[1] + line[3], line[4]
        test.append(tail)
        test_rel.append(rel)
        mask_type_list.append('tail')
        test.append(head)
        test_rel.append(rel)
        mask_type_list.append('head')

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

    coke_config = init_bert_net_config(args, logger, print_config=True)
    if args.bmtrain:
        model = BertBaseBMT(config=coke_config)
        bmt.init_parameters(model)
        bmt.load(model, args.ckpt_root)
        bmt.synchronize()
    else:
        model = BertBase(config=coke_config)
        model.tokenizer = test_dataset.tokenizer

        if args.gpus > 1:
            model = nn.DataParallel(model, device_ids=device_ids)
        state_dict = torch.load(args.ckpt_path, map_location=device)
        for name, param in state_dict.items():
            if "linear2" in name:
                print(param)
        model.load_state_dict(state_dict)
        model.to(device=device)

    # ------------
    # evaluation
    # ------------
    model.eval()
    print("----------- evaluation -----------")
    topk_entity_list = []
    realtime_hits10, realtime_hits10_rate = 0.0, 0.0
    realtime_hits3, realtime_hits3_rate = 0.0, 0.0
    realtime_hits1, realtime_hits1_rate = 0.0, 0.0
    realtime_mr, realtime_mr_rate = 0.0, 0.0
    realtime_mrr, realtime_mrr_rate = 0.0, 0.0
    count = 0
    mem_total_acc = 0.0
    for iter, batch_data in tqdm(enumerate(test_data_loader)):
        # fetch batch data
        try:
            src_ids, input_mask, seg_ids, pos_ids, mem_label_ids, mem_label_idx, mask_types = batch_data
        except RuntimeError:
            print("One data instance's length should be 5, received {}.".format(len(batch_data)))
            continue
        if args.use_cuda:
            src_ids, input_mask, seg_ids, pos_ids, mem_label_idx, mem_label_ids = \
                src_ids.to(device), input_mask.to(device), seg_ids.to(device), \
                pos_ids.to(device), mem_label_idx.to(device), mem_label_ids.to(device)
        mem_probs_list = []
        for j in range(args.batch_size):
            temp_seg = seg_ids[j]
            temp_pos = pos_ids[j]
            temp_src = src_ids[j]
            temp_src[mem_label_idx[j][0]:mem_label_idx[j][1]+1] = 0
            temp_input_mask = input_mask[j]
            temp_input_mask[mem_label_idx[j][0]:mem_label_idx[j][1]+1] = 0
            temp_prob_list = []
            for i in range(18):
                temp_src[mem_label_idx[j][0]: mem_label_idx[j][0]+i+1] = test_dataset.mask_id
                try:
                    temp_src[mem_label_idx[j][0]+i+1] = test_dataset.ent_end_id
                except:
                    print(temp_src)
                temp_input_mask[mem_label_idx[j][0]: mem_label_idx[j][0]+i+2] = 1

                input_x = {
                    'src_ids': temp_src.unsqueeze(dim=0),
                    'input_mask': temp_input_mask.unsqueeze(dim=0),
                    'segment_ids': temp_seg.unsqueeze(dim=0),
                    'position_ids': temp_pos.unsqueeze(dim=0)
                }
                # forward
                mem_probs = model(input_x)["logits"]
                mem_probs = list(mem_probs.cpu().detach().numpy())
                temp_prob_list.append(mem_probs)
            mem_probs_list.append(temp_prob_list)

        mem_label_idx = mem_label_idx.cpu().detach().numpy()
        mem_label_ids = mem_label_ids.cpu().detach().numpy()
        mem_probs_list = np.array(mem_probs_list)

        for j in range(args.batch_size):
            target = mem_label_ids[j][mem_label_idx[j][0]: mem_label_idx[j][1]]
            test_target = np.array(entity2token_id[np.array([test[(iter * args.batch_size) + (j)]])[0]])
            assert (target == test_target).all()

            filter_set = set()
            target_entity = test[(iter * args.batch_size) + (j)]
            target_rel = test_rel[(iter * args.batch_size) + (j)]
            entity_prob_list = []
            for idx, _entity_id in enumerate(range(len(id2entity))):
                entity = id2entity[_entity_id]
                if entity in filter_dict[target_entity][target_rel] and entity != target_entity:
                    filter_set.add(_entity_id)
                _entity_token_id = entity2token_id[entity]
                entity_len = len(_entity_token_id)
                preds = mem_probs_list[j][entity_len - 1][0][mem_label_idx[j][0]: mem_label_idx[j][0] + entity_len]
                _entity_prob = preds[np.arange(entity_len), _entity_token_id]
                entity_prob_list.append(np.mean(_entity_prob))

            topk_entity_idx = np.argsort(entity_prob_list)[::-1]  # [:10]
            topk_entity = []
            for id in topk_entity_idx:
                if id not in filter_set:
                    topk_entity.append(id2entity[id])

            topk_entity_list.append(topk_entity)

            test_list = []
            test_list.append(topk_entity)
            test_array = np.array(test_list)
            target = np.array([test[(iter * args.batch_size) + (j)]])
            test_hits1, test_hits3, test_hits10, test_MR, test_MRR = eval_fn(target, test_array, realtime_mr_rate,
                                                                             realtime_mrr_rate)

            realtime_hits10 += test_hits10
            realtime_hits3 += test_hits3
            realtime_hits1 += test_hits1
            realtime_mrr += test_MRR
            realtime_mr += test_MR
            count += 1
            # print(test_MRR)
            # print("#################")
        realtime_hits10_rate = realtime_hits10 / count
        realtime_hits3_rate = realtime_hits3 / count
        realtime_hits1_rate = realtime_hits1 / count
        realtime_mr_rate = realtime_mr / count
        realtime_mrr_rate = realtime_mrr / count
        print("realtime_hits1_rate:", realtime_hits1_rate)
        print("realtime_hits3_rate", realtime_hits3_rate)
        print("realtime_hits10_rate", realtime_hits10_rate)
        print("realtime_mrr_rate", realtime_mrr_rate)
        print("realtime_mr_rate:", realtime_mr_rate)

    # topk_entity_array = np.array(topk_entity_list)
    # test_hits1, test_hits3, test_hits10, test_MR, test_MRR = eval_fn(test, topk_entity_array, realtime_mr_rate, realtime_mrr_rate)
    # print(test_hits1)
    # print(test_hits3)
    # print(test_hits10)
    # print(test_MR)
    # print(test_MRR)

    eval_result = {
        'hits1': realtime_hits1_rate,
        'hits3': realtime_hits3_rate,
        'hits10': realtime_hits10_rate,
        'test_MR': realtime_mr_rate,
        'test_MRR': realtime_mrr_rate
    }
    with open(args.save_path + 'eval_result.json', "w") as fw:
        fw.write(json.dumps(eval_result, indent=4) + "\n")


if __name__ == '__main__':
    main()