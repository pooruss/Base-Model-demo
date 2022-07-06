""" data preprocess
"""
from torch.utils.data import Dataset
import os
import csv
import sys
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


def mask_tokens(vocab_size, tokenizer, inputs, mlm_prob):
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
    labels[~masked_indices] = 0  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    """
    对于mask的数据,其中80%是赋值为MASK.
    这里先对所有数据以0.8概率获取伯努利分布值, 
    然后 和maksed_indices 进行与操作,得到Mask 的80%的概率 indice, 对这些位置赋值为MASK 
    """
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # print(inputs_tensor)
    inputs_tensor[indices_replaced] = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
    # print(inputs_tensor)
    # 10% of the time, we replace masked input tokens with random word
    """
    对于mask_indices剩下的20% 在进行提取,取其中一半进行random 赋值,剩下一般保留原来值. 
    """
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(999, vocab_size, labels.shape, dtype=torch.long)
    inputs_tensor[indices_random] = random_words[indices_random]
    return list(np.array(inputs_tensor)), list(np.array(labels))


class MLMFeatures(object):
    """A single set of features of data."""

    def __init__(self, src_ids, segment_ids, position_ids, input_mask, mlm_label_ids,
                 mem_label_ids, mem_label_idx):
        self.src_ids = src_ids
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.input_mask = input_mask
        self.mlm_label_ids = mlm_label_ids
        self.mem_label_ids = mem_label_ids
        self.mem_label_idx = mem_label_idx


class MEMFeatures(object):
    """A single set of features of data."""

    def __init__(self, src_ids, segment_ids, position_ids, input_mask, mem_label_ids):
        self.src_ids = src_ids
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.input_mask = input_mask
        self.mem_label_ids = mem_label_ids


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

    def __init__(self, tokenizer, max_entity_len=16, max_rel_len=8, max_mlm_seq_len=512):
        self.labels = set()
        self.tokenizer = tokenizer
        self.max_len = max_mlm_seq_len
        self.max_entity_len = max_entity_len
        self.max_rel_len = max_rel_len
        self.cls_id = [self.tokenizer.convert_tokens_to_ids("[CLS]")]
        self.sep_id = [self.tokenizer.convert_tokens_to_ids("[SEP]")]
        self.ent_begin_id = [self.tokenizer.convert_tokens_to_ids("[EB]")]
        self.ent_end_id = [self.tokenizer.convert_tokens_to_ids("[EE]")]
        self.rel_begin_id = [self.tokenizer.convert_tokens_to_ids("[RB]")]
        self.rel_end_id = [self.tokenizer.convert_tokens_to_ids("[RE]")]
        self.pad_id = [self.tokenizer.convert_tokens_to_ids("[PAD]")]
        self.mask_id = [self.tokenizer.convert_tokens_to_ids("[EMASK]")]
        self.mask_id = [self.tokenizer.convert_tokens_to_ids("[MASK]")]

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_with_path.tsv")), "train", "text", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", "text", data_dir)

    def get_alias_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_with_path.tsv")), "train", "alias", data_dir)

    def get_alias_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", "alias", data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", "text", data_dir)

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

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_type, data_dir):
        """Creates examples for the training and dev sets."""
        features_file = os.path.join(data_dir, set_type + "_" + data_type + "_features.npy")
        features_path = Path(features_file)
        if features_path.exists():
            features = np.load(features_file, allow_pickle=True)
            return features

        # entity to text
        ent2alias = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    ent2alias[temp[0]] = temp[1]  # [:end]
        ent2text = {}
        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r', encoding='utf-8') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    # first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]
        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r', encoding='utf-8') as f:
            rel_lines = f.readlines()
            for line in rel_lines:
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]
        relations = list(rel2text.keys())
        # lines_str_set = set(['\t'.join(line) for line in lines])

        examples, mlm_features = [], []

        if data_type == 'alias':
            ent2text = ent2alias

        if set_type == 'test' or set_type == 'dev':
            text_type_list = ['head']
        else:
            text_type_list = ['tail', 'head', 'mid']
        for text_type in text_type_list:
            for (index, line) in tqdm(enumerate(lines)):
                if index % 10000 == 0:
                    logger.info("Writing example %d of %d" % (index, len(lines)))
                item_nums = len(line)
                if item_nums == 3:
                    max_text_len = self.max_len - 7 - self.max_entity_len * 2 - self.max_rel_len
                    head, rel, tail = line
                    relation_text = rel2text[rel]
                    head_alias = ent2alias[head]
                    tail_alias = ent2alias[tail]
                    if text_type == 'tail':
                        try:
                            tail_text = ent2text[tail]
                        except:
                            print("Cannot find description for entity {}!".format(tail))
                            continue
                        rel_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(relation_text))
                        ha_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(head_alias))
                        ta_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail_alias))
                        tt_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail_text))
                        tt_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                               tokenizer=self.tokenizer,
                                                               inputs=tt_tokens,
                                                               mlm_prob=0.15)
                        # ha_tokens = self.mask_id * len(ha_tokens)
                        # all_tokens = self.cls_id + self.ent_begin_id + ha_tokens + self.ent_end_id + \
                        #              self.rel_begin_id + rel_tokens + self.rel_end_id + \
                        #              self.ent_begin_id + ta_tokens + tt_tokens + self.ent_end_id
                        # _truncate_ids(all_tokens, len(all_tokens), self.max_len)
                        # mem_label_ids = [0] * 2 + ha_tokens
                        # _truncate_ids(mem_label_ids, len(mem_label_ids), self.max_len)
                        # mem_label_idx = [2, len(ha_tokens) + 2]
                        # mlm_label_ids = [0] * (len(self.cls_id + self.ent_begin_id + ha_tokens + self.ent_end_id +
                        #                            self.rel_begin_id + rel_tokens + self.rel_end_id +
                        #                            self.ent_begin_id + ta_tokens)) + mlm_label_ids
                        # _truncate_ids(mlm_label_ids, len(mlm_label_ids), self.max_len)

                        _truncate_pad_sequence((ha_tokens, ta_tokens, rel_tokens, tt_tokens), text_type,
                                               self.max_entity_len, self.max_rel_len, max_text_len, self.max_len - 7)
                        mem_label_ids = [0] * 2 + ha_tokens + [0] * (self.max_len - self.max_entity_len - 2)
                        ha_tokens = self.mask_id * len(ha_tokens)
                        mem_label_idx = [2, self.max_entity_len + 2]
                        _truncate_ids(mlm_label_ids, len(mlm_label_ids), max_text_len)
                        mlm_label_ids = [0] * (self.max_len - max_text_len - 1) + mlm_label_ids + [0]
                        all_tokens = self.cls_id + self.ent_begin_id + ha_tokens + self.ent_end_id + \
                                     self.rel_begin_id + rel_tokens + self.rel_end_id + \
                                     self.ent_begin_id + ta_tokens + tt_tokens + self.ent_end_id
                        head_len, tail_len = len(ha_tokens), len(ta_tokens + tt_tokens)
                    elif text_type == 'head':
                        try:
                            head_text = ent2text[head]
                        except:
                            print("Cannot find description for entity {}!".format(head))
                            continue
                        rel_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(relation_text))
                        ha_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(head_alias))
                        ta_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail_alias))
                        ht_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(head_text))
                        ht_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                               tokenizer=self.tokenizer,
                                                               inputs=ht_tokens,
                                                               mlm_prob=0.15)
                        # ta_tokens = self.mask_id * len(ta_tokens)
                        # all_tokens = self.cls_id + self.ent_begin_id + ha_tokens + ht_tokens + self.ent_end_id + \
                        #              self.rel_begin_id + rel_tokens + self.rel_end_id + \
                        #              self.ent_begin_id + ta_tokens + self.ent_end_id
                        # _truncate_ids(all_tokens, len(all_tokens), self.max_len)
                        # temp_len = len(
                        #     self.cls_id + self.ent_begin_id + ha_tokens + ht_tokens + self.ent_end_id + \
                        #     self.rel_begin_id + rel_tokens + self.rel_end_id + \
                        #     self.ent_begin_id)
                        # mem_label_ids = [0] * temp_len + ta_tokens
                        # mem_label_idx = [temp_len, len(ta_tokens) + temp_len]
                        # _truncate_ids(mem_label_ids, len(mem_label_ids), self.max_len)
                        # mlm_label_ids = [0] * len(self.cls_id + self.ent_begin_id + ha_tokens) + mlm_label_ids
                        # _truncate_ids(mlm_label_ids, len(mlm_label_ids), self.max_len)

                        _truncate_pad_sequence((ha_tokens, ta_tokens, rel_tokens, ht_tokens), text_type,
                                               self.max_entity_len, self.max_rel_len, max_text_len, self.max_len - 7)
                        mem_label_ids = [0] * (self.max_len - self.max_entity_len - 1) + ta_tokens + [0]
                        mem_label_idx = [self.max_len - 2 - self.max_entity_len, self.max_len - 2]
                        ta_tokens = self.mask_id * len(ta_tokens)
                        _truncate_ids(mlm_label_ids, len(mlm_label_ids), max_text_len)
                        mlm_label_ids = [0] * (2 + self.max_entity_len) + mlm_label_ids + [0] * (5 + self.max_rel_len + self.max_entity_len)
                        all_tokens = self.cls_id + self.ent_begin_id + ha_tokens + ht_tokens + self.ent_end_id + \
                                     self.rel_begin_id + rel_tokens + self.rel_end_id + \
                                     self.ent_begin_id + ta_tokens + self.ent_end_id
                        head_len, tail_len = len(ha_tokens + ht_tokens), len(ta_tokens)
                    else:
                        continue
                    all_len, rel_len = len(all_tokens), len(rel_tokens)

                    # segment id
                    segment_ids = [0] * (head_len + 3) + [1] * (rel_len + 2) + [0] * (tail_len + 2)
                    _truncate_ids(segment_ids, len(segment_ids), self.max_len)

                    # position id
                    position_ids = []
                    for i in range(all_len):
                        position_ids += [i]

                    # input mask
                    input_mask = []
                    for i in range(all_len):
                        input_mask += [1 if all_tokens[i] != 0 else 0]

                    all_tokens = np.array(all_tokens)
                    segment_ids = np.array(segment_ids)
                    position_ids = np.array(position_ids)
                    input_mask = np.array(input_mask)
                    mlm_label_ids = np.array(mlm_label_ids)
                    mem_label_ids = np.array(mem_label_ids)

                    assert len(all_tokens) == self.max_len
                    assert len(segment_ids) == self.max_len
                    assert len(position_ids) == self.max_len
                    assert len(input_mask) == self.max_len
                    assert len(mlm_label_ids) == self.max_len
                    assert len(mem_label_ids) == self.max_len

                    if index == 0:
                        print("############ Triple input data#############")
                        print("input tokens:{}".format(self.tokenizer.convert_ids_to_tokens(all_tokens)))
                        print("input ids:{}".format(all_tokens))
                        print("input masks:{}".format(input_mask))
                        print("segment ids:{}".format(segment_ids))
                        print("position ids:{}".format(position_ids))
                        print("mlm label:{}".format(mlm_label_ids))
                        print("mem label:{}".format(mem_label_ids))
                        print("mem label idx:{}".format(mem_label_idx))

                    mlm_feature = MLMFeatures(
                        src_ids=all_tokens,
                        segment_ids=segment_ids,
                        position_ids=position_ids,
                        input_mask=input_mask,
                        mlm_label_ids=mlm_label_ids,
                        mem_label_ids=mem_label_ids,
                        mem_label_idx=mem_label_idx)
                    mlm_features.append(mlm_feature)

                elif item_nums == 5:
                    max_text_len = self.max_len - 11 - self.max_entity_len * 2 - self.max_rel_len * 2
                    relation_1_text = rel2text[line[1]]
                    relation_2_text = rel2text[line[3]]
                    rel_1_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(relation_1_text))
                    rel_2_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(relation_2_text))
                    if text_type == 'head':
                        try:
                            head = ent2text[line[0]]
                        except:
                            print("Cannot find description for entity {}!".format(ent2alias[line[0]]))
                            continue
                        mid, tail = ent2alias[line[2]], ent2alias[line[4]]
                        head_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(head))
                        mid_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(mid))
                        tail_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail))
                        head_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                                 tokenizer=self.tokenizer,
                                                                 inputs=head_tokens,
                                                                 mlm_prob=0.15)

                        _truncate_ids(mlm_label_ids, len(mlm_label_ids), max_text_len)
                        mlm_label_ids = [0] * 2 + mlm_label_ids + [0] * (self.max_len - 2 - max_text_len)
                        _truncate_pad_sequence((head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens, tail_tokens),
                                               text_type, self.max_entity_len, self.max_rel_len, max_text_len,
                                               self.max_len - 11)
                        head_tokens = self.cls_id + self.ent_begin_id + head_tokens + self.ent_end_id
                        mem_label_ids = [0] * (self.max_len - self.max_entity_len - 1) + tail_tokens + [0]
                        mem_label_idx = [self.max_len - 2 - self.max_entity_len, self.max_len - 2]
                        tail_tokens = self.ent_begin_id + self.mask_id * len(tail_tokens) + self.ent_end_id

                        rel_1_tokens = self.rel_begin_id + rel_1_tokens + self.rel_end_id
                        mid_tokens = self.ent_begin_id + mid_tokens + self.ent_end_id
                        rel_2_tokens = self.rel_begin_id + rel_2_tokens + self.rel_end_id
                        all_tokens = rel_1_tokens + mid_tokens + rel_2_tokens
                        all_tokens = head_tokens + all_tokens + tail_tokens
                        # input_mask = (text_input_mask + [1] * (len(all_tokens) - len(head_tokens)))

                    elif text_type == 'mid':
                        try:
                            mid = ent2text[line[2]]
                        except:
                            print("Cannot find description for entity {}!".format(ent2alias[line[2]]))
                            continue
                        head, tail = ent2alias[line[0]], ent2alias[line[4]]
                        head_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(head))
                        mid_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(mid))
                        tail_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail))
                        mid_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                                tokenizer=self.tokenizer,
                                                                inputs=mid_tokens,
                                                                mlm_prob=0.15)
                        _truncate_pad_sequence((head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens, tail_tokens),
                                               text_type, self.max_entity_len, self.max_rel_len, max_text_len,
                                               self.max_len - 11)
                        _truncate_ids(mlm_label_ids, len(mlm_label_ids), max_text_len)
                        mlm_label_ids = [0] * (self.max_entity_len + 6 + self.max_rel_len) + mlm_label_ids + [0] * (
                                    self.max_entity_len + self.max_rel_len + 5)
                        mid_tokens = self.ent_begin_id + mid_tokens + self.ent_end_id
                        rnd = random.random()
                        if rnd <= 0.5:
                            mask_type = 'tail'
                            mem_label_ids = [0] * (self.max_len - self.max_entity_len - 1) + tail_tokens + [0]
                            mem_label_idx = [self.max_len - 2 - self.max_entity_len, self.max_len - 2]

                            tail_tokens = self.ent_begin_id + self.mask_id * len(tail_tokens) + self.ent_end_id
                            head_tokens = self.cls_id + self.ent_begin_id + head_tokens + self.ent_end_id

                            rel_1_tokens = self.rel_begin_id + rel_1_tokens + self.rel_end_id
                            rel_2_tokens = self.rel_begin_id + rel_2_tokens + self.rel_end_id
                            all_tokens = self.tokenizer.convert_tokens_to_ids(head_tokens + rel_1_tokens) + mid_tokens \
                                         + self.tokenizer.convert_tokens_to_ids(rel_2_tokens) + tail_tokens
                        else:
                            mask_type = 'head'
                            mem_label_ids = [0] * 2 + head_tokens + [0] * (self.max_len - 2 - self.max_entity_len)
                            mem_label_idx = [2, self.max_entity_len + 2]
                            head_tokens = self.cls_id + self.ent_begin_id + \
                                          self.mask_id * len(head_tokens) + self.ent_end_id
                            tail_tokens = self.ent_begin_id + tail_tokens + self.ent_end_id
                            rel_1_tokens = self.rel_begin_id + rel_1_tokens + self.rel_end_id
                            rel_2_tokens = self.rel_begin_id + rel_2_tokens + self.rel_end_id
                            all_tokens = head_tokens + rel_1_tokens + mid_tokens + rel_2_tokens + tail_tokens
                        # input_mask = ([1] * (head_len + rel_1_len) + text_input_mask + [1] * (rel_2_len + tail_len))
                    else:
                        try:
                            tail = ent2text[line[4]]
                        except:
                            print("Cannot find description for entity {}!".format(ent2alias[line[4]]))
                            continue
                        head, mid = ent2alias[line[0]], ent2alias[line[2]]
                        head_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(head))
                        mid_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(mid))
                        tail_tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(tail))
                        tail_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                                 tokenizer=self.tokenizer,
                                                                 inputs=tail_tokens,
                                                                 mlm_prob=0.15)
                        _truncate_pad_sequence((head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens, tail_tokens),
                                               text_type, self.max_entity_len, self.max_rel_len, max_text_len,
                                               self.max_len - 11)
                        _truncate_ids(mlm_label_ids, len(mlm_label_ids), max_text_len)
                        mlm_label_ids = [0] * (self.max_len - max_text_len - 1) + mlm_label_ids + [0]
                        tail_tokens = self.ent_begin_id + tail_tokens + self.ent_end_id

                        mem_label_ids = [0] * 2 + head_tokens + [0] * (self.max_len - 2 - self.max_entity_len)
                        mem_label_idx = [2, self.max_entity_len + 2]

                        head_tokens = self.cls_id + self.ent_begin_id + \
                                      self.mask_id * len(head_tokens) + self.ent_end_id

                        rel_1_tokens = self.rel_begin_id + rel_1_tokens + self.rel_end_id
                        mid_tokens = self.ent_begin_id + mid_tokens + self.ent_end_id
                        rel_2_tokens = self.rel_begin_id + rel_2_tokens + self.rel_end_id
                        all_tokens = rel_1_tokens + mid_tokens + rel_2_tokens
                        all_tokens = head_tokens + all_tokens + tail_tokens
                        # input_mask = ([1] * (len(all_tokens) - len(tail_tokens)) + text_input_mask)

                    all_len, head_len, rel_1_len, mid_len, rel_2_len, tail_len = \
                        len(all_tokens), len(head_tokens), len(rel_1_tokens), len(mid_tokens), len(rel_2_tokens), len(
                            tail_tokens)

                    # segment id
                    segment_ids = [0] * head_len + [1] * rel_1_len + [0] * mid_len + [1] * rel_2_len + [0] * tail_len

                    # position id
                    position_ids = []
                    for i in range(all_len):
                        position_ids += [i]

                    # input mask
                    input_mask = []
                    for i in range(len(all_tokens)):
                        input_mask += [1 if all_tokens[i] != 0 else 0]

                    all_tokens = np.array(all_tokens)
                    segment_ids = np.array(segment_ids)
                    position_ids = np.array(position_ids)
                    input_mask = np.array(input_mask)
                    mlm_label_ids = np.array(mlm_label_ids)
                    mem_label_ids = np.array(mem_label_ids)
                    assert len(all_tokens) == self.max_len
                    assert len(segment_ids) == self.max_len
                    assert len(position_ids) == self.max_len
                    assert len(input_mask) == self.max_len
                    assert len(mlm_label_ids) == self.max_len
                    assert len(mem_label_ids) == self.max_len
                    assert len(mem_label_idx) == 2
                    # assert len(mem_mask_pos) == len(mem_label_ids)
                    if len(mlm_features) == 0:
                        print("############ Path input data#############")
                        print("input tokens:{}".format(self.tokenizer.convert_ids_to_tokens(all_tokens)))
                        print("input ids:{}".format(all_tokens))
                        print("input masks:{}".format(input_mask))
                        print("segment ids:{}".format(segment_ids))
                        print("position ids:{}".format(position_ids))
                        print("mlm label:{}".format(mlm_label_ids))
                        print("mem label:{}".format(mem_label_ids))
                        print("mem label idx:{}".format(mem_label_idx))
                    mlm_feature = MLMFeatures(
                        src_ids=all_tokens,
                        segment_ids=segment_ids,
                        position_ids=position_ids,
                        input_mask=input_mask,
                        mlm_label_ids=mlm_label_ids,
                        mem_label_ids=mem_label_ids,
                        mem_label_idx=mem_label_idx)
                    mlm_features.append(mlm_feature)
        np.save(features_file, np.array(mlm_features))
        return mlm_features


def _truncate_ids(input_ids, ids_len, max_ids_len):
    if ids_len <= max_ids_len:
        input_ids.extend((max_ids_len - ids_len) * [0])
    else:
        while True:
            input_ids.pop()
            if len(input_ids) <= max_ids_len:
                break


def _truncate_entity(input_tokens, entity_len, max_entity_len):
    if entity_len <= max_entity_len:
        input_tokens.extend((max_entity_len - entity_len) * [0])
    else:
        while True:
            input_tokens.pop()
            if len(input_tokens) <= max_entity_len:
                break


def _truncate_relation(input_tokens, rel_len, max_rel_len):
    if rel_len <= max_rel_len:
        input_tokens.extend((max_rel_len - rel_len) * [0])
    else:
        while True:
            input_tokens.pop()
            if len(input_tokens) <= max_rel_len:
                break


def _truncate_text(input_tokens, text_len, max_text_len):
    if text_len <= max_text_len:
        input_tokens.extend((max_text_len - text_len) * [0])
    else:
        while True:
            input_tokens.pop()
            if len(input_tokens) <= max_text_len:
                break


def _truncate_pad_sequence(input_tokens, text_type, max_entity_len, max_rel_len, max_text_len, max_len):
    if len(input_tokens) == 4:
        head_tokens, tail_tokens, rel_tokens, text_tokens = input_tokens
        head_len, rel_len, tail_len, text_len = len(head_tokens), len(rel_tokens), len(tail_tokens), len(text_tokens)

        _truncate_relation(rel_tokens, rel_len, max_rel_len)
        _truncate_entity(head_tokens, head_len, max_entity_len)
        _truncate_entity(tail_tokens, tail_len, max_entity_len)
        if text_type == 'head':
            _truncate_text(text_tokens, text_len, max_text_len)
        elif text_type == 'tail':
            _truncate_text(text_tokens, text_len, max_text_len)

    elif len(input_tokens) == 5:
        head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens, tail_tokens = input_tokens
        head_len, rel_1_len, mid_len, rel_2_len, tail_len = \
            len(head_tokens), len(rel_1_tokens), len(mid_tokens), len(rel_2_tokens), len(tail_tokens)

        _truncate_relation(rel_1_tokens, rel_1_len, max_rel_len)
        _truncate_relation(rel_2_tokens, rel_2_len, max_rel_len)

        if text_type == 'head':
            _truncate_entity(mid_tokens, mid_len, max_entity_len)
            _truncate_entity(tail_tokens, tail_len, max_entity_len)
            _truncate_text(head_tokens, head_len, max_text_len)

        elif text_type == 'mid':
            _truncate_entity(head_tokens, head_len, max_entity_len)
            _truncate_entity(tail_tokens, tail_len, max_entity_len)
            _truncate_text(mid_tokens, mid_len, max_text_len)

        elif text_type == 'tail':
            _truncate_entity(mid_tokens, mid_len, max_entity_len)
            _truncate_entity(head_tokens, head_len, max_entity_len)
            _truncate_text(tail_tokens, tail_len, max_text_len)
