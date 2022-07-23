""" data preprocess
"""
from doctest import testfile
from operator import neg
from torch.utils.data import Dataset
import os
import csv
import sys
import json
import collections
import logging
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

torch.manual_seed(1)


def do_mask_ids(vocab_size, tokenizer, inputs, mlm_prob, mask_id):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = torch.from_numpy(np.array(inputs)).clone()
    inputs_tensor = torch.from_numpy(np.array(inputs)).clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults
    # to 0.15 in Bert/RoBERTa)
    """
    prob_data = torch.full(labels.shape, args.mlm_probability) 
        是生成一个labels一样大小的矩阵,里面的值默认是0.15.
    torch.bernoulli(prob_data),从伯努利分布中抽取二元随机数(0或者1),
        prob_data是上面产生的是一个所有值为0.15(在0和1之间的数),
        输出张量的第i个元素值,将以输入张量的第i个概率值等于1.
        (在这里 即输出张量的每个元素有 0.15的概率为1, 0.85的概率为0. 15%原始数据 被mask住)
    """
    masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_prob)).bool()
    """
    mask_indices通过bool()函数转成True,False
    下面对于85%原始数据 没有被mask的位置进行赋值为-1
    """
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    """
    对于mask的数据,其中80%是赋值为MASK.
    这里先对所有数据以0.8概率获取伯努利分布值, 
    然后 和maksed_indices 进行与操作,得到Mask 的80%的概率 indice, 对这些位置赋值为MASK 
    """
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # print(inputs_tensor)
    inputs_tensor[indices_replaced] = mask_id
    inputs_tensor = inputs_tensor.long()
    # print(inputs_tensor)
    # 10% of the time, we replace masked input tokens with random word
    """
    对于mask_indices剩下的20% 在进行提取,取其中一半进行random 赋值,剩下一般保留原来值. 
    """
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs_tensor[indices_random] = random_words[indices_random]
    return list(np.array(inputs_tensor)), list(np.array(labels))


class MLERMFeatures(object):
    """A single set of features of data."""

    def __init__(self, src_ids, input_mask, mlm_label_ids,
                 mem_label_ids, mem_label_idx, pos_neg_seq, margin_labels, mask_type):
        self.src_ids = src_ids
        self.input_mask = input_mask
        self.mlm_label_ids = mlm_label_ids
        self.mem_label_ids = mem_label_ids
        self.mem_label_idx = mem_label_idx
        self.pos_neg_seq = pos_neg_seq
        self.margin_labels = margin_labels
        self.mask_type = mask_type


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            f.close()
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self, data_dir, tokenizer, max_entity_len=16, max_rel_len=8, max_mlm_seq_len=512, neg_nums=3,
                 ignore=True, save_dir=''):
        self.labels = set()
        self.tokenizer = tokenizer
        self.max_len = max_mlm_seq_len
        self.max_entity_len = max_entity_len
        self.max_rel_len = max_rel_len

        self.cls_id = self.tokenizer.convert_tokens_to_ids("<s>")
        self.sep_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.ent_begin_id = self.tokenizer.convert_tokens_to_ids("<ent>")
        self.ent_end_id = self.tokenizer.convert_tokens_to_ids("</ent>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("<mask>")
        self.unk_id = self.tokenizer.convert_tokens_to_ids("<unk>")
        self.tl_id = self.pad_id
        self.tr_id = self.mask_id
        self.neg_nums = neg_nums
        self.ignore = ignore
        self.data_dir = data_dir
        self.save_dir = save_dir

        self.true_triples_dict = {}
        self.entity_len_dict = {}
        # self.triples_dict = self.get_triples_dict(self.data_dir)
        # self.entity_frequency = self.get_entity_frequency(self.data_dir)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", "text", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", "text", data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test-N.tsv")), "test", "text", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    # 正例实体dict
    def get_true_triples_dict(self, all_triples):
        true_triples_dict = {"forward":{}, "backward":{}}
        for triple in all_triples:
            head, rel, tail = triple
            if head not in true_triples_dict["forward"]:
                true_triples_dict["forward"][head] = {}
            if rel not in true_triples_dict["forward"][head]:
                true_triples_dict["forward"][head][rel] = set()
            true_triples_dict["forward"][head][rel].add(tail)

            if tail not in true_triples_dict["backward"]:
                true_triples_dict["backward"][tail] = {}
            if rel not in true_triples_dict["backward"][tail]:
                true_triples_dict["backward"][tail][rel] = set()
            true_triples_dict["backward"][tail][rel].add(head)
        self.true_triples_dict = true_triples_dict

    # 实体长度dict
    def get_entity_len_dict(self, head_entities, tail_entities, ent2alias):
        entity_len_dict = {"head":{}, "tail":{}}
        for head, tail in zip(head_entities, tail_entities):
            entity = head
            entity_name = ent2alias[entity]
            entity_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity_name))
            entity_len = len(entity_tokens)
            if entity_len not in entity_len_dict["head"]:
                entity_len_dict["head"][entity_len] = {}
            if entity not in entity_len_dict["head"][entity_len]:
                entity_len_dict["head"][entity_len][entity] = 0
            entity_len_dict["head"][entity_len][entity] += 1

            entity = tail
            entity_name = ent2alias[entity]
            entity_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(entity_name))
            entity_len = len(entity_tokens)
            if entity_len not in entity_len_dict["tail"]:
                entity_len_dict["tail"][entity_len] = {}
            if entity not in entity_len_dict["tail"][entity_len]:
                entity_len_dict["tail"][entity_len][entity] = 0
            entity_len_dict["tail"][entity_len][entity] += 1

        for i in range(self.max_entity_len + 1):
            if i == 0:
                continue
            if i in entity_len_dict['head']:
                head_sorted_list = sorted(entity_len_dict['head'][i].items(), key=lambda x: x[1], reverse=True)
                entity_len_dict['head'][i] = head_sorted_list
            if i in entity_len_dict['tail']:
                tail_sorted_list = sorted(entity_len_dict['tail'][i].items(), key=lambda x: x[1], reverse=True)
                entity_len_dict['tail'][i] = tail_sorted_list
        self.entity_len_dict = entity_len_dict

    def get_negatives(self, triple, entity_ids, text_type, neg_nums):
        head, rel, tail = triple
        if text_type == "head":
            entity = head
            entity_type = "forward"
            entity_pos = "tail"
        else:
            entity = tail
            entity_type = "backward"
            entity_pos = "head"

        entity_len = len(entity_ids)

        negatives = []
        if text_type == "head":
            entity_type = "forward"
        else:
            entity_type = "backward"

        # 取 token 长度一样的实体集
        entity_list = self.entity_len_dict[entity_pos][entity_len]
        for centity_cnt in entity_list:
            centity, cnt = centity_cnt
            if entity in self.true_triples_dict[entity_type]:
                if rel in self.true_triples_dict[entity_type][entity]:
                    # 过滤true实体
                    if centity not in self.true_triples_dict[entity_type][entity][rel]:
                        negatives.append(centity)
                        if len(negatives) == neg_nums:
                            break
        return negatives

    def _create_examples(self, lines, set_type, data_type, data_dir):
        """Creates examples for the training and dev sets."""
        features_file = os.path.join(self.save_dir, set_type + "_" + data_type + "_features.npy")
        features_path = Path(features_file)
        if self.ignore:
            if features_path.exists():
                features = np.load(features_file, allow_pickle=True)
                return features

        # entity to text
        ent2alias = {}
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    if data_dir.find("FB15") != -1:
                        ent2alias[temp[0]] = temp[1]
                    elif data_dir.find("WN18") != -1:
                        end = temp[1].find(',')
                        ent2alias[temp[0]] = temp[1][:end]
                        ent2text[temp[0]] = temp[1][end:]

        max_desc_len = 0.0
        if data_dir.find("FB15") != -1:
            ent2text = {}
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r', encoding='utf-8') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1][:first_sent_end_position + 1]  # [:first_sent_end_position + 1]
                    token_len = len(self.tokenizer.tokenize(temp[1][:first_sent_end_position + 1]))
                    if token_len > max_desc_len:
                        max_desc_len = token_len
        # print(max_desc_len)

        rel_file = open(os.path.join(data_dir, "relation2template.json"))
        rel2text = json.load(rel_file)

        all_triples = self._read_tsv(os.path.join(data_dir, "all.tsv"))
        head_entities = []
        for triple in all_triples:
            head, rel, tail = triple
            head_entities.append(head)
        tail_entities = []
        for triple in all_triples:
            head, rel, tail = triple
            tail_entities.append(tail)
        self.get_true_triples_dict(all_triples)
        self.get_entity_len_dict(head_entities, tail_entities, ent2alias)


        # relations = list(rel2text.keys())
        # lines_str_set = set(['\t'.join(line) for line in lines])

        # max_ent_len, max_rel_len1, max_text_len = 0.0, 0.0, 0.0
        # for ent in ent2alias:
        #     token = self.tokenizer.tokenize(ent2alias[ent])
        #     if len(token) > max_ent_len:
        #         max_ent_len = len(token)
        # for rel in rel2text:
        #     token = self.tokenizer.tokenize(rel2text[rel])
        #     if len(token) > max_rel_len1:
        #         max_rel_len1 = len(token)
        # for ent in ent2text:
        #     token = self.tokenizer.tokenize(ent2text[ent])
        #     if len(token) > max_text_len:
        #         max_text_len = len(token)
        #         print(token)
        # print("###########")
        # print(max_ent_len)
        # print(max_rel_len1)
        # print(max_text_len)
        # print("###########")
        # x = input()

        empty_entity = set()
        examples, mlm_features = [], []
        ori_max_rel_len = self.max_rel_len
        path_max_rel_len = self.max_rel_len * 2 + self.max_entity_len
        if 'alias' in data_type:
            ent2text = ent2alias

        test_file = open(os.path.join(data_dir, "demo_test.txt"), 'w', encoding='utf-8')
        for (index, line) in tqdm(enumerate(lines)):
            if index % 10000 == 0:
                logger.info("Writing example %d of %d" % (index, len(lines)))
            item_nums = len(line)
            if set_type == 'test' or set_type == 'dev':
                text_type_list = ['head', 'tail']
            else:
                text_type_list = ['head', 'tail']
            for text_type in text_type_list:
                negatives = []
                pos_neg_seq = []
                margin_labels = []
                if item_nums == 3:
                    is_triple = True
                    head, rel, tail = line
                    self.max_rel_len = ori_max_rel_len
                    max_text_len = self.max_len - 5 - self.max_entity_len * 2 - self.max_rel_len
                    relation_text = rel2text[rel]
                    head_alias = ent2alias[head]
                    tail_alias = ent2alias[tail]
                else:
                    is_triple = False
                    head, rel1, mid, rel2, tail = line[0], line[1], line[2], line[3], line[4]
                    self.max_rel_len = path_max_rel_len
                    max_text_len = self.max_len - 7 - self.max_entity_len * 2 - self.max_rel_len
                    relation_text = rel2text[rel1] + rel2text[rel2]
                    head_alias = ent2alias[head]
                    tail_alias = ent2alias[tail]

                if tail in ent2text:
                    tail_text = ent2text[tail]
                else:
                    if tail not in empty_entity:
                        print("Cannot find description for entity {}!".format(tail))
                        empty_entity.add(tail)
                    tail_text = ent2alias[tail]
                if head in ent2text:
                    head_text = ent2text[head]
                else:
                    if head not in empty_entity:
                        print("Cannot find description for entity {}!".format(head))
                        empty_entity.add(head)
                    head_text = ent2alias[head]

                # 替换为关系模板，构造三元组文本
                this_template = relation_text.replace('[X]', '::;;##').replace('[Y]', '::;;##')
                prompts = this_template.split('::;;##')
                prompts = [x.strip() for x in prompts]
                assert(len(prompts) == 3)
                idx_x = relation_text.find('[X]')
                idx_y = relation_text.find('[Y]')
                if idx_x < idx_y:
                    final_list = [prompts[0], "<s>", head_alias.strip(), "</s>", prompts[1], "<pad>", tail_alias.strip(), "<mask>", prompts[2]]
                else:
                    final_list = [prompts[0], "<pad>", tail_alias.strip(), "<mask>", prompts[1], "<s>", head_alias.strip(), "</s>", prompts[2]]

                triple_sent = ''.join(final_list).strip()

                triple_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(triple_sent))
                ha_pos, hae_pos = triple_ids.index(self.cls_id), triple_ids.index(self.sep_id)
                triple_ids[ha_pos] = self.ent_begin_id
                triple_ids[hae_pos] = self.ent_end_id

                ta_pos, tae_pos = triple_ids.index(self.tl_id), triple_ids.index(self.tr_id)
                triple_ids[ta_pos] = self.ent_begin_id
                triple_ids[tae_pos] = self.ent_end_id

                ha_ids = triple_ids[ha_pos + 1: hae_pos]
                # print(self.tokenizer.convert_ids_to_tokens(ha_ids))
                # print(head_alias)
                ta_ids = triple_ids[ta_pos + 1: tae_pos]
                # print(self.tokenizer.convert_ids_to_tokens(ta_ids))
                # print(tail_alias)
                ht_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(head_text))
                # ta_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail_alias))
                tt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail_text))

                if text_type == 'head':
                    mask_pos = ta_pos + 1
                    mask_end_pos = tae_pos
                    mask_ids = ta_ids
                    text_ids = ht_ids
                    mask_type = "tail"
                else:
                    mask_pos = ha_pos + 1
                    mask_end_pos = hae_pos
                    mask_ids = ha_ids
                    text_ids = tt_ids
                    mask_type = "head"

                negatives = self.get_negatives(line, mask_ids, text_type, self.neg_nums)
                assert len(negatives) == self.neg_nums
                for i in range(len(mask_ids)):
                    triple_ids[mask_pos + i] = self.mask_id

                if len(text_ids) > max_text_len:
                    _truncate_ids(text_ids, max_text_len)
                if set_type == 'train' or set_type == 'dev':
                    # do mlm
                    text_ids, mlm_label_ids = do_mask_ids(vocab_size=len(self.tokenizer.get_vocab()),
                                                           tokenizer=self.tokenizer,
                                                           inputs=text_ids,
                                                           mlm_prob=0.2, mask_id=self.mask_id)
                else:
                    mlm_label_ids = text_ids

                # old template
                input_seq = [self.cls_id] + text_ids
                mlm_label_ids = [self.pad_id] + mlm_label_ids
                _truncate_ids(mlm_label_ids, self.max_len)

                mem_label_ids = [self.pad_id] * len(input_seq) + [self.pad_id] * mask_pos + mask_ids
                _truncate_ids(mem_label_ids, self.max_len)
                mem_label_idx = [len(input_seq) + mask_pos, len(input_seq) + mask_pos + len(mask_ids)]

                pos_triple_ids = triple_ids[:mask_pos] + mask_ids + triple_ids[mask_end_pos:]
                pos_seq = input_seq + pos_triple_ids
                _truncate_ids(pos_seq, self.max_len)
                assert len(pos_seq) == self.max_len
                pos_neg_seq.append(pos_seq)
                margin_labels += [1] * len(mask_ids)
                # corrupt head and tail
                for neg_entity in negatives:
                    neg_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ent2alias[neg_entity]))
                    assert len(neg_ids) == len(mask_ids)
                    neg_triple_ids = triple_ids[:mask_pos] + neg_ids + triple_ids[mask_end_pos:]
                    assert len(neg_triple_ids) == len(triple_ids)
                    neg_seq = input_seq + neg_triple_ids
                    _truncate_ids(neg_seq, self.max_len)
                    pos_neg_seq.append(neg_seq)
                    margin_labels += [0] * len(neg_ids)

                _truncate_ids(margin_labels, self.max_len)

                input_seq = input_seq + triple_ids
                _truncate_ids(input_seq, self.max_len)

                # # new template
                # if text_type == 'head':
                #     input_seq = [self.cls_id] + text_ids
                #     mlm_label_ids = [self.pad_id] + mlm_label_ids
                #     _truncate_ids(mlm_label_ids, self.max_len)

                #     mem_label_ids = [self.pad_id] * len(input_seq) + mem_label_ids
                #     _truncate_ids(mem_label_ids, self.max_len)
                #     mem_label_idx = [len(input_seq) + mask_pos, len(input_seq) + mask_pos + len(mask_ids)]

                #     input_seq = input_seq + triple_ids
                #     _truncate_ids(input_seq, self.max_len)
                # else:
                #     input_seq = [self.cls_id] + triple_ids
                #     mlm_label_ids = [self.pad_id] * len(input_seq) + mlm_label_ids
                #     _truncate_ids(mlm_label_ids, self.max_len)

                #     mem_label_ids = [self.pad_id] + mem_label_ids
                #     _truncate_ids(mem_label_ids, self.max_len)
                #     mem_label_idx = [1 + mask_pos, 1 + mask_pos + len(mask_ids)]

                #     input_seq = input_seq + text_ids
                #     _truncate_ids(input_seq, self.max_len)

                all_len = len(input_seq)
                # input mask
                input_mask = []
                for i in range(all_len):
                    input_mask += [1 if input_seq[i] != self.pad_id else 0]

                all_tokens = np.array(input_seq)
                input_mask = np.array(input_mask)
                mlm_label_ids = np.array(mlm_label_ids)
                mem_label_ids = np.array(mem_label_ids)
                assert len(all_tokens) == self.max_len
                assert len(input_mask) == self.max_len
                assert len(mlm_label_ids) == self.max_len
                assert len(mem_label_ids) == self.max_len
                assert len(pos_neg_seq) == (self.neg_nums + 1)
                if index == 0:
                    test_file.write(triple_sent + '\n')
                    test_file.write("############ Triple input data#############" + '\n')
                    test_file.write("input tokens:{}".format(self.tokenizer.convert_ids_to_tokens(all_tokens)) + '\n')
                    test_file.write("input ids:{}".format(all_tokens) + '\n')
                    test_file.write("input masks:{}".format(input_mask) + '\n')
                    test_file.write("mlm label:{}".format(mlm_label_ids) + '\n')
                    test_file.write("mem label:{}".format(mem_label_ids) + '\n')
                    test_file.write("mem label idx:{}".format(mem_label_idx) + '\n')
                    test_file.flush()
                    print(triple_sent)
                    print("############ Triple input data#############")
                    print("input tokens:{}".format(self.tokenizer.convert_ids_to_tokens(all_tokens)))
                    print("input ids:{}".format(all_tokens))
                    print("input masks:{}".format(input_mask))
                    print("mlm label:{}".format(mlm_label_ids))
                    print("mem label:{}".format(mem_label_ids))
                    print("mem label idx:{}".format(mem_label_idx))

                mlm_feature = MLERMFeatures(
                    src_ids=all_tokens,
                    input_mask=input_mask,
                    mlm_label_ids=mlm_label_ids,
                    mem_label_ids=mem_label_ids,
                    mem_label_idx=mem_label_idx,
                    pos_neg_seq=pos_neg_seq,
                    mask_type=mask_type)
                mlm_features.append(mlm_feature)

        np.save(features_file, np.array(mlm_features))
        empty_file = open(os.path.join(data_dir, set_type + "_" + data_type + "_empty_entities.txt"), 'w', encoding='utf-8')
        for entity in empty_entity:
            empty_file.write(entity + "\n")
        return mlm_features


def _truncate_ids(input_ids, max_ids_len):
    ids_len = len(input_ids)
    if ids_len <= max_ids_len:
        input_ids.extend((max_ids_len - ids_len) * [1])
    else:
        while True:
            input_ids.pop()
            if len(input_ids) <= max_ids_len:
                break


if __name__ == '__main__':
    from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
    import sys

    max_entity_len = int(sys.argv[1])  # FB15K237 21 ; WN18RR 17
    max_rel_len = int(sys.argv[2])  # FB15K237 23 ; WN18RR 5
    max_seq_len = int(sys.argv[3])  # FB15K237 187 ; WN18RR 192
    task = sys.argv[4]
    task_name = sys.argv[5]
    data_dir = sys.argv[6]
    save_dir = sys.argv[7]

    tokenizer = RobertaTokenizer.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/roberta-base")
    processor = KGProcessor(data_dir, tokenizer, max_entity_len, max_rel_len, max_seq_len, 3, False, save_dir=save_dir)

    features = []
    if task == 'train':
            features = processor.get_train_examples(data_dir)
    elif task == 'dev':
            features = processor.get_dev_examples(data_dir)
    elif task == 'test':
            features = processor.get_test_examples(data_dir)
