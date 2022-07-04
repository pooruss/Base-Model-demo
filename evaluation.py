import os
import json
import numpy as np
from tqdm import tqdm
import collections
import torch
from torch.utils.data import DataLoader
from models.bigmodel import CoKE, CoKE_Roberta, CoKE_BMT
from data.coke_dataset import KBCDataset
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
model_g.add_arg("hidden_size", int, 256, "CoKE model config: hidden size, default 256")
model_g.add_arg("num_hidden_layers", int, 12, "CoKE model config: num_hidden_layers, default 12")
model_g.add_arg("num_attention_heads", int, 4, "CoKE model config: num_attention_heads, default 4")
# model_g.add_arg("vocab_size", int, 16396, "CoKE model config: vocab_size")
model_g.add_arg("num_relations", int, None, "CoKE model config: vocab_size")
model_g.add_arg("max_position_embeddings", int, 40, "CoKE model config: max_position_embeddings")
model_g.add_arg("dropout", float, 0.2, "CoKE model config: dropout, default 0.1")
model_g.add_arg("hidden_dropout", float, 0.2, "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("attention_dropout", float, 0.2,
                "CoKE model config: attention_probs_dropout_prob, default 0.1")
model_g.add_arg("initializer_range", int, 0.02, "CoKE model config: initializer_range")
model_g.add_arg("intermediate_size", int, 512, "CoKE model config: intermediate_size, default 512")
model_g.add_arg("weight_sharing", bool, True, "If set, share weights between word embedding and masked lm.")

train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg("gpus", int, 4, "GPU nums.")
train_g.add_arg("node", int, 1, "Node nums.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("dataset", str, "FB15K", "dataset name")
# data_g.add_arg("data_root", str, './data/FB15K/', "Vocab path.")
data_g.add_arg("vocab_file", str, 'vocab.txt', "Vocab path.")
data_g.add_arg("true_triple_file", str, 'all.txt', "All triple data for coke.")
data_g.add_arg("sen_candli_file", str, 'sen_candli.txt',
               "sentence_candicate_list file for path query evaluation. Only used for path query datasets")
data_g.add_arg("sen_trivial_file", str, 'trival_sen.txt',
               "trivial sentence file for pathquery evaluation. Only used for path query datasets")

parser.add_argument("--task_name", default='path', type=str, required=True, help="path or triple.")
parser.add_argument("--data_root", default='./data/FB15K/', type=str, required=True, help="data directory.")
parser.add_argument("--vocab_size", default=16396, type=int, required=True, help="16396 for fb15k, 75169 for pathFB.")
parser.add_argument("--max_seq_len", default=3, type=int, required=True, help="sequence length.")
parser.add_argument("--use_cuda", default=False, type=bool, required=True, help="gpu or cpu.")
parser.add_argument("--gpu_ids", default='0', type=str, required=True, help="gpu ids.")
parser.add_argument("--batch_size", default=2048, type=int, required=True, help="batch size.")
parser.add_argument("--test_file", default='test.coke.txt', type=str, required=True, help="test file.")
parser.add_argument("--checkpoint", default='./model.pt', type=str, required=True, help="ckpt path.")
parser.add_argument("--save_path", default='./checkpoints/', type=str, required=True, help="result path.")
parser.add_argument("--model_name", default='coke', type=str, required=True, help="coke or coke_roberta.")
parser.add_argument("--use_ema", default=True, type=boolean_string, help="use ema or not.")
parser.add_argument("--bmtrain", default=True, type=boolean_string, help="use bmtrain or not.")

args = parser.parse_args()


def load_eval_dict(true_triple_file):
    def load_true_triples(true_triple_file):
        true_triples = []
        with open(true_triple_file, "r") as fr:
            for line in fr.readlines():
                tokens = line.strip("\r \n").split("\t")
                assert len(tokens) == 3
                true_triples.append(
                    (int(tokens[0]), int(tokens[1]), int(tokens[2])))
        logger.debug("Finish loading %d true triples" % len(true_triples))
        return true_triples

    true_triples = load_true_triples(true_triple_file)
    true_triples_dict = collections.defaultdict(lambda: {'hs': collections.defaultdict(list),
                                                         'ts': collections.defaultdict(list)})
    for h, r, t in true_triples:
        true_triples_dict[r]['ts'][h].append(t)
        true_triples_dict[r]['hs'][t].append(h)
    return true_triples_dict


def batch_evaluation(eval_i, all_examples, batch_results, true_triples):
    r_hts_idx = collections.defaultdict(list)
    scores_head = collections.defaultdict(list)
    scores_tail = collections.defaultdict(list)
    batch_r_hts_cnt = 0
    b_size = len(batch_results)
    for j in range(b_size):
        result = batch_results[j]
        i = eval_i + j
        example = all_examples[i]
        assert len(example.token_ids
                   ) == 3, "For kbc task each example consists of 3 tokens"
        h, r, t = example.token_ids

        mask_type = example.mask_type
        if i % 2 == 0:
            r_hts_idx[r].append((h, t))
            batch_r_hts_cnt += 1
        if mask_type == "MASK_HEAD":
            scores_head[(r, t)] = result
        elif mask_type == "MASK_TAIL":
            scores_tail[(r, h)] = result
        else:
            raise ValueError("Unknown mask type in prediction example:%d" % i)

    rank = {}
    f_rank = {}
    for r, hts in r_hts_idx.items():
        r_rank = {'head': [], 'tail': []}
        r_f_rank = {'head': [], 'tail': []}
        for h, t in hts:
            scores_t = scores_tail[(r, h)][:]
            sortidx_t = np.argsort(scores_t)[::-1]
            r_rank['tail'].append(np.where(sortidx_t == t)[0][0] + 1)

            rm_idx = true_triples[r]['ts'][h]
            rm_idx = [i for i in rm_idx if i != t]
            for i in rm_idx:
                scores_t[i] = -np.Inf
            sortidx_t = np.argsort(scores_t)[::-1]
            r_f_rank['tail'].append(np.where(sortidx_t == t)[0][0] + 1)

            scores_h = scores_head[(r, t)][:]
            sortidx_h = np.argsort(scores_h)[::-1]
            r_rank['head'].append(np.where(sortidx_h == h)[0][0] + 1)

            rm_idx = true_triples[r]['hs'][t]
            rm_idx = [i for i in rm_idx if i != h]
            for i in rm_idx:
                scores_h[i] = -np.Inf
            sortidx_h = np.argsort(scores_h)[::-1]
            r_f_rank['head'].append(np.where(sortidx_h == h)[0][0] + 1)
        rank[r] = r_rank
        f_rank[r] = r_f_rank

    h_pos = [p for k in rank.keys() for p in rank[k]['head']]
    t_pos = [p for k in rank.keys() for p in rank[k]['tail']]
    f_h_pos = [p for k in f_rank.keys() for p in f_rank[k]['head']]
    f_t_pos = [p for k in f_rank.keys() for p in f_rank[k]['tail']]

    ranks = np.asarray(h_pos + t_pos)
    f_ranks = np.asarray(f_h_pos + f_t_pos)
    return ranks, f_ranks


def compute_metrics(rank_li, frank_li, output_evaluation_result_file):
    """ combine the kbc rank results from batches into the final metrics """
    rank_rets = np.array(rank_li).ravel()
    frank_rets = np.array(frank_li).ravel()
    mrr = np.mean(1.0 / rank_rets)
    fmrr = np.mean(1.0 / frank_rets)

    hits1 = np.mean(rank_rets <= 1.0)
    hits3 = np.mean(rank_rets <= 3.0)
    hits10 = np.mean(rank_rets <= 10.0)
    # filtered metrics
    fhits1 = np.mean(frank_rets <= 1.0)
    fhits3 = np.mean(frank_rets <= 3.0)
    fhits10 = np.mean(frank_rets <= 10.0)

    eval_result = {
        'mrr': mrr,
        'hits1': hits1,
        'hits3': hits3,
        'hits10': hits10,
        'fmrr': fmrr,
        'fhits1': fhits1,
        'fhits3': fhits3,
        'fhits10': fhits10
    }
    with open(output_evaluation_result_file, "w") as fw:
        fw.write(json.dumps(eval_result, indent=4) + "\n")
    return eval_result


def main():
    if args.bmtrain:
        bmt.init_distributed(seed=0)
    # ------------
    # data
    # ------------
    args.valid_data_path = os.path.join(args.data_root, args.test_file)
    args.true_triple_path = os.path.join(args.data_root, args.true_triple_file)
    args.vocab_path = os.path.join(args.data_root, args.vocab_file)
    val_dataset = KBCDataset(vocab_path=args.vocab_path,
                             data_path=args.valid_data_path,
                             max_seq_len=args.max_seq_len,
                             vocab_size=args.vocab_size)
    if args.bmtrain:
        val_data_loader = DistributedDataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    true_triplets_dict = load_eval_dict(args.true_triple_path)

    # ------------
    # model
    # ------------
    device_ids = list()
    for gpu_id in args.gpu_ids:
        device_ids.append(int(gpu_id))
    device = torch.device('cuda:{}'.format(device_ids[0]) if args.use_cuda else 'cpu')
    # device = torch.device('cpu')
    args.gpus = len(device_ids)
    coke_config = init_coke_net_config(args, logger, print_config=True)
    if args.model_name == 'coke_roberta':
        model = CoKE_Roberta(config=coke_config)
    elif args.bmtrain:
        model = CoKE_BMT(config=coke_config)
        bmt.load(model, args.checkpoint)
    else:
        model = CoKE(config=coke_config)
        # load checkpoint
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device=device)
        for name, param in model.named_parameters():
            print("name:{}".format(name))
            print(param)

    # ------------
    # evaluation
    # ------------
    # model.eval()
    print("----------- evaluation -----------")
    eval_i = 0
    batch_eval_rets = []
    f_batch_eval_rets = []
    for iter, batch_data in tqdm(enumerate(val_data_loader)):
        # fetch batch data
        try:
            src_id, pos_id, input_mask, mask_pos, mask_label = batch_data
        except RuntimeError:
            print("Per data instance's length should be 5, received {}.".format(len(batch_data)))
            continue
        if args.use_cuda:
            src_id, pos_id, input_mask, mask_pos, mask_label = \
                src_id.to(device), pos_id.to(device), input_mask.to(device), mask_pos.to(device), mask_label.to(device)
        input_x = {
            'src_ids': src_id,
            'position_ids': pos_id,
            'input_mask': input_mask,
            'mask_pos': mask_pos,
            'mask_label': mask_label
        }
        # forward
        batch_results = model(input_x)['logits'].detach().cpu().numpy()

        rank, frank = batch_evaluation(eval_i, val_dataset.examples, batch_results, true_triplets_dict)
        eval_i += src_id.size(0)
        batch_eval_rets.extend(rank)
        f_batch_eval_rets.extend(frank)

    eval_result_file = os.path.join(args.save_path, "eval_result.json")
    eval_result = compute_metrics(batch_eval_rets, f_batch_eval_rets, eval_result_file)

    outs = "%.3f\t%.3f\t%.3f\t%.3f" % (
        eval_result['fmrr'], eval_result['fhits1'], eval_result['fhits3'], eval_result['fhits10'])
    logger.info("\n----------- Evaluation Performance --------------\n%s\n%s" %
                ("\t".join(["TASK", "MRR", "Hits@1", "Hits@3", "Hits@10"]), outs))


if __name__ == '__main__':
    main()
