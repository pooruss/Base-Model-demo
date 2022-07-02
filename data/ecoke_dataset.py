""" data reader for CoKE
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
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel, RobertaTokenizer, RobertaConfig, BertTokenizer
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
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
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


class PathInputExample3(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, text_d=None, text_e=None, text_type=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.text_d = text_d
        self.text_e = text_e
        self.text_type = text_type
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, all_ids, input_mask, segment_ids, position_ids, cls_label):
        self.all_ids = all_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.cls_label = cls_label


class MLMFeatures(object):
    """A single set of features of data."""

    def __init__(self, src_ids, segment_ids, position_ids, input_mask, mlm_label_ids,
                 mem_label_ids, mem_mask_pos):
        self.src_ids = src_ids
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.input_mask = input_mask
        self.mlm_label_ids = mlm_label_ids
        self.mem_label_ids = mem_label_ids
        self.mem_mask_pos = mem_mask_pos


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, mask_id, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, mask_id, data_dir):
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

    def __init__(self, tokenizer, max_mlm_seq_len):
        self.labels = set()
        self.tokenizer = tokenizer
        self.max_len = max_mlm_seq_len
        self.cls_id = [self.tokenizer.convert_tokens_to_ids("[CLS]")]
        self.sep_id = [self.tokenizer.convert_tokens_to_ids("[SEP]")]
        self.ent_begin_id = [self.tokenizer.convert_tokens_to_ids("[EB]")]
        self.ent_end_id = [self.tokenizer.convert_tokens_to_ids("[EE]")]
        self.rel_begin_id = [self.tokenizer.convert_tokens_to_ids("[RB]")]
        self.rel_end_id = [self.tokenizer.convert_tokens_to_ids("[RE]")]
        self.pad_id = [self.tokenizer.convert_tokens_to_ids("[PAD]")]
        self.e_mask_id = [self.tokenizer.convert_tokens_to_ids("[EMASK]")]
        self.mask_id = [self.tokenizer.convert_tokens_to_ids("[MASK]")]

    def get_train_examples(self, emask_id, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_path.tsv")), "train", emask_id, data_dir)

    def get_dev_examples(self, emask_id, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_path.tsv")), "dev", emask_id, data_dir)

    def get_test_examples(self, emask_id, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_path.tsv")), "test", emask_id, data_dir)

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

    def _create_examples(self, lines, set_type, emask_id, data_dir):
        """Creates examples for the training and dev sets."""
        features_file = os.path.join(data_dir, set_type + "_features.npy")
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

        # lines_str_set = set(['\t'.join(line) for line in lines])
        examples, mlm_features = [], []
        for (index, line) in tqdm(enumerate(lines)):
            if index % 10000 == 0:
                logger.info("Writing example %d of %d" % (index, len(lines)))
            relation_1_text = rel2text[line[1]]
            relation_2_text = rel2text[line[3]]
            text_type_list = ["head", "mid", "tail"]
            if set_type == "dev" or set_type == "test":
                label = "1"
                self.labels.add(label)
            elif set_type == "train":
                for text_type in text_type_list:
                    rel_1_tokens = self.tokenizer.tokenize(relation_1_text)
                    rel_2_tokens = self.tokenizer.tokenize(relation_2_text)
                    if text_type == 'head':
                        try:
                            head = ent2text[line[0]]
                        except:
                            print("Cannot find description for entity {}!".format(ent2alias[line[0]]))
                            continue
                        mid, tail = ent2alias[line[2]], ent2alias[line[4]]
                        head_tokens = self.tokenizer.tokenize(head)
                        mid_tokens = self.tokenizer.tokenize(mid)
                        tail_tokens = self.tokenizer.tokenize(tail)
                        text_input_mask = _truncate_pad_sequence(head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens,
                                                                 tail_tokens, text_type, self.max_len - 11)
                        head_tokens = self.tokenizer.convert_tokens_to_ids(head_tokens)
                        head_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                                 tokenizer=self.tokenizer,
                                                                 inputs=head_tokens,
                                                                 mlm_prob=0.15)
                        mlm_label_ids = [-1] * 2 + mlm_label_ids + [-1] * (self.max_len - len(mlm_label_ids) - 2)
                        head_tokens = self.cls_id + self.ent_begin_id + head_tokens + self.ent_end_id
                        mem_label_ids = [-1] * 495 + self.tokenizer.convert_tokens_to_ids(tail_tokens) + [-1]

                        tail_tokens = self.ent_begin_id + self.e_mask_id * len(tail_tokens) + self.ent_end_id

                        rel_1_tokens = ["[RB]"] + rel_1_tokens + ["[RE]"]
                        mid_tokens = ["[EB]"] + mid_tokens + ["[EE]"]
                        rel_2_tokens = ["[RB]"] + rel_2_tokens + ["[RE]"]

                        all_tokens = rel_1_tokens + mid_tokens + rel_2_tokens
                        all_tokens = self.tokenizer.convert_tokens_to_ids(all_tokens)
                        all_tokens = head_tokens + all_tokens + tail_tokens
                        # input_mask = (text_input_mask + [1] * (len(all_tokens) - len(head_tokens)))

                    elif text_type == 'mid':
                        try:
                            mid = ent2text[line[2]]
                        except:
                            print("Cannot find description for entity {}!".format(ent2alias[line[2]]))
                            continue
                        head, tail = ent2alias[line[0]], ent2alias[line[4]]
                        head_tokens = self.tokenizer.tokenize(head)
                        mid_tokens = self.tokenizer.tokenize(mid)
                        tail_tokens = self.tokenizer.tokenize(tail)
                        text_input_mask = _truncate_pad_sequence(head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens,
                                                                 tail_tokens, text_type, self.max_len - 11)
                        mid_tokens = self.tokenizer.convert_tokens_to_ids(mid_tokens)
                        mid_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                                 tokenizer=self.tokenizer,
                                                                 inputs=mid_tokens,
                                                                 mlm_prob=0.15)
                        mlm_label_ids = [-1] * 30 + mlm_label_ids + [-1] * 29
                        mid_tokens = self.ent_begin_id + mid_tokens + self.ent_end_id
                        rnd = random.random()
                        if rnd <= 0.5:
                            mask_type = 'tail'
                            mem_label_ids = [-1] * 495 + self.tokenizer.convert_tokens_to_ids(tail_tokens) + [-1]
                            tail_tokens = self.ent_begin_id + self.e_mask_id * len(tail_tokens) + self.ent_end_id
                            head_tokens = ["[CLS]"] + ["[EB]"] + head_tokens + ["[EE]"]

                            rel_1_tokens = ["[RB]"] + rel_1_tokens + ["[RE]"]
                            rel_2_tokens = ["[RB]"] + rel_2_tokens + ["[RE]"]
                            all_tokens = self.tokenizer.convert_tokens_to_ids(head_tokens + rel_1_tokens) + mid_tokens \
                                         + self.tokenizer.convert_tokens_to_ids(rel_2_tokens) + tail_tokens
                        else:
                            mask_type = 'head'
                            mem_label_ids = [-1] * 2 + self.tokenizer.convert_tokens_to_ids(head_tokens) + [-1] * 494
                            head_tokens = self.cls_id + self.ent_begin_id + \
                                          self.e_mask_id * len(head_tokens) + self.ent_end_id
                            tail_tokens = ["[EB]"] + tail_tokens + ["[EE]"]
                            rel_1_tokens = ["[RB]"] + rel_1_tokens + ["[RE]"]
                            rel_2_tokens = ["[RB]"] + rel_2_tokens + ["[RE]"]
                            all_tokens = head_tokens + self.tokenizer.convert_tokens_to_ids(rel_1_tokens) + mid_tokens \
                                         + self.tokenizer.convert_tokens_to_ids(rel_2_tokens + tail_tokens)
                        head_len, rel_1_len, rel_2_len, tail_len = \
                            len(head_tokens), len(rel_1_tokens), len(rel_2_tokens), len(tail_tokens)
                        # input_mask = ([1] * (head_len + rel_1_len) + text_input_mask + [1] * (rel_2_len + tail_len))
                    else:
                        try:
                            tail = ent2text[line[4]]
                        except:
                            print("Cannot find description for entity {}!".format(ent2alias[line[4]]))
                            continue
                        head, mid = ent2alias[line[0]], ent2alias[line[2]]
                        head_tokens = self.tokenizer.tokenize(head)
                        mid_tokens = self.tokenizer.tokenize(mid)
                        tail_tokens = self.tokenizer.tokenize(tail)
                        text_input_mask = _truncate_pad_sequence(head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens,
                                                                 tail_tokens, text_type, self.max_len - 11)
                        tail_tokens = self.tokenizer.convert_tokens_to_ids(tail_tokens)
                        tail_tokens, mlm_label_ids = mask_tokens(vocab_size=len(self.tokenizer.get_vocab()),
                                                                 tokenizer=self.tokenizer,
                                                                 inputs=tail_tokens,
                                                                 mlm_prob=0.15)
                        mlm_label_ids = [-1] * (self.max_len - len(mlm_label_ids) - 1) + mlm_label_ids + [-1] * 1
                        tail_tokens = self.ent_begin_id + tail_tokens + self.ent_end_id

                        mem_label_ids = [-1] * 2 + self.tokenizer.convert_tokens_to_ids(head_tokens) + [-1] * 494

                        head_tokens = self.cls_id + self.ent_begin_id + \
                                      self.e_mask_id * len(head_tokens) + self.ent_end_id

                        rel_1_tokens = ["[RB]"] + rel_1_tokens + ["[RE]"]
                        mid_tokens = ["[EB]"] + mid_tokens + ["[EE]"]
                        rel_2_tokens = ["[RB]"] + rel_2_tokens + ["[RE]"]
                        all_tokens = rel_1_tokens + mid_tokens + rel_2_tokens
                        all_tokens = self.tokenizer.convert_tokens_to_ids(all_tokens)
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
                    mem_mask_pos = np.argwhere(all_tokens.reshape(-1) == self.e_mask_id)
                    mem_label_ids = mem_label_ids.reshape(-1, 1)[mem_mask_pos]

                    # print(len(mem_mask_pos))
                    # if index % 10000 == 0:
                    #     print("head")
                    #     print(head)
                    #     print("mid")
                    #     print(mid)
                    #     print("tail")
                    #     print(tail)
                    #     print("all_tokens")
                    #     print(all_tokens)
                    #     print("segment_ids")
                    #     print(segment_ids)
                    #     print("position_ids")
                    #     print(position_ids)
                    #     print("input_mask")
                    #     print(input_mask)
                    #     print("mlm_label_ids")
                    #     print(mlm_label_ids)
                    #     print("mem_label_ids")
                    #     print(mem_label_ids)
                    #     print("mem_mask_pos")
                    #     print(mem_mask_pos)

                    assert len(all_tokens) == self.max_len
                    assert len(segment_ids) == self.max_len
                    assert len(position_ids) == self.max_len
                    assert len(input_mask) == self.max_len
                    assert len(mlm_label_ids) == self.max_len
                    assert len(mem_label_ids) == 16
                    assert len(mem_mask_pos) == len(mem_label_ids)

                    mlm_feature = MLMFeatures(
                        src_ids=all_tokens,
                        segment_ids=segment_ids,
                        position_ids=position_ids,
                        input_mask=input_mask,
                        mlm_label_ids=mlm_label_ids,
                        mem_label_ids=mem_label_ids,
                        mem_mask_pos=mem_mask_pos)
                    mlm_features.append(mlm_feature)
                # rnd = random.random()
                # guid = "%s-%s" % (set_type + "_corrupt", i)
                # if rnd <= 0.5:
                #     # corrupting head
                #     for j in range(2):
                #         tmp_head = ''
                #         while True:
                #             tmp_ent_list = set(entities)
                #             tmp_ent_list.remove(line[0])
                #             tmp_ent_list = list(tmp_ent_list)
                #             tmp_head = random.choice(tmp_ent_list)
                #             tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + line[4]
                #             if tmp_triple_str not in lines_str_set:
                #                 break
                #         tmp_head_text = ent2text[tmp_head]
                #         examples.append(
                #             PathInputExample3(guid=guid, text_a=tmp_head_text, text_b=relation_1_text, text_c=tail_ent_text, label="0"))
                # else:
                #     # corrupting tail
                #     tmp_tail = ''
                #     for j in range(2):
                #         while True:
                #             tmp_ent_list = set(entities)
                #             tmp_ent_list.remove(line[2])
                #             tmp_ent_list = list(tmp_ent_list)
                #             tmp_tail = random.choice(tmp_ent_list)
                #             tmp_triple_str = line[0] + '\t' + line[1] + '\t' + line[2] + '\t' + line[3] + '\t' + tmp_tail
                #             if tmp_triple_str not in lines_str_set:
                #                 break
                #         tmp_tail_text = ent2text[tmp_tail]
                #         examples.append(
                #             PathInputExample3(guid=guid, text_a=head_ent_text, text_b=relation_1_text, text_c=tmp_tail_text, label="0"))
        np.save(features_file, np.array(mlm_features))
        return mlm_features


def _truncate_pad_sequence(head_tokens, rel_1_tokens, mid_tokens, rel_2_tokens, tail_tokens, text_type, max_len):
    head_len, rel_1_len, mid_len, rel_2_len, tail_len = \
        len(head_tokens), len(rel_1_tokens), len(mid_tokens), len(rel_2_tokens), len(tail_tokens)
    if rel_1_len <= 8:
        rel_1_tokens.extend((8 - rel_1_len) * ["[PAD]"])
    else:
        while True:
            rel_1_tokens.pop()
            if len(rel_1_tokens) <= 8:
                break
    if rel_2_len <= 8:
        rel_2_tokens.extend((8 - rel_2_len) * ["[PAD]"])
    else:
        while True:
            rel_2_tokens.pop()
            if len(rel_2_tokens) <= 8:
                break

    input_mask = [1]
    if text_type == 'head':
        if mid_len <= 16:
            mid_tokens.extend((16 - mid_len) * ["[PAD]"])
        else:
            while True:
                mid_tokens.pop()
                if len(mid_tokens) <= 16:
                    break
        if tail_len <= 16:
            tail_tokens.extend((16 - tail_len) * ["[PAD]"])
        else:
            while True:
                tail_tokens.pop()
                if len(tail_tokens) <= 16:
                    break
        input_mask += [1]
        if head_len > max_len - 48:
            while True:
                head_tokens.pop()
                head_len -= 1
                if head_len <= max_len - 48:
                    break
            for i in range(max_len - 48 + 1):
                input_mask += [1]
        else:
            head_tokens.extend((max_len - 48 - head_len) * ["[PAD]"])
            for i in range(len(head_tokens)):
                input_mask += [0 if head_tokens[i] == "[PAD]" else 1]
            input_mask += [1]

    if text_type == 'mid':
        if head_len <= 16:
            head_tokens.extend((16 - head_len) * ["[PAD]"])
        else:
            while True:
                head_tokens.pop()
                if len(head_tokens) <= 16:
                    break
        if tail_len <= 16:
            tail_tokens.extend((16 - tail_len) * ["[PAD]"])
        else:
            while True:
                tail_tokens.pop()
                if len(tail_tokens) <= 16:
                    break
        if mid_len > max_len - 48:
            while True:
                mid_tokens.pop()
                mid_len -= 1
                if mid_len <= max_len - 48:
                    break
            for i in range(max_len - 48 + 1):
                input_mask += [1]
        else:
            mid_tokens.extend((max_len - 48 - mid_len) * ["[PAD]"])
            for i in range(len(mid_tokens)):
                input_mask += [0 if mid_tokens[i] == "[PAD]" else 1]
            input_mask += [1]
        assert len(head_tokens) == 16
        assert len(head_tokens) == 16

    if text_type == 'tail':
        if mid_len <= 16:
            mid_tokens.extend((16 - mid_len) * ["[PAD]"])
        else:
            while True:
                mid_tokens.pop()
                if len(mid_tokens) <= 16:
                    break
        if head_len <= 16:
            head_tokens.extend((16 - head_len) * ["[PAD]"])
        else:
            while True:
                head_tokens.pop()
                if len(head_tokens) <= 16:
                    break
        if tail_len > max_len - 48:
            while True:
                tail_tokens.pop()
                tail_len -= 1
                if tail_len <= max_len - 48:
                    break
            for i in range(max_len - 48 + 1):
                input_mask += [1]
        else:
            tail_tokens.extend((max_len - 48 - tail_len) * ["[PAD]"])
            for i in range(len(tail_tokens)):
                input_mask += [0 if tail_tokens[i] == "[PAD]" else 1]
            input_mask += [1]

    return input_mask


class KBCDataset(Dataset):
    """ DataReader
    """

    def __init__(self, data_dir, do_train, do_eval, do_test, max_seq_len=3, vocab_size=-1):
        # D:/Study/BMKG/git_clone/BMKG/bmkg/base_model/
        self.tokenizer = BertTokenizer.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        print(self.vocab_size)
        self.sep_id = self.tokenizer.convert_tokens_to_ids("[SEP]")
        self.ent_begin_id = self.tokenizer.convert_tokens_to_ids("[EB]")
        self.ent_end_id = self.tokenizer.convert_tokens_to_ids("[EE]")
        self.rel_begin_id = self.tokenizer.convert_tokens_to_ids("[RB]")
        self.rel_end_id = self.tokenizer.convert_tokens_to_ids("[RE]")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        self.e_mask_id = self.tokenizer.convert_tokens_to_ids("[EMASK]")
        # if vocab_size > 0:
        #     assert len(self.vocab) == vocab_size, \
        #         "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
        #         (vocab_size, len(self.vocab))
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.max_seq_len = max_seq_len

        self.max_length = 512

        self.features = self.read_example_features(data_dir)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        return torch.tensor(feature.src_ids, dtype=torch.long), \
               torch.tensor(feature.input_mask, dtype=torch.long), \
               torch.tensor(feature.segment_ids, dtype=torch.long), \
               torch.tensor(feature.position_ids, dtype=torch.long), \
               torch.tensor(feature.mlm_label_ids, dtype=torch.long), \
               torch.tensor(feature.mem_label_ids, dtype=torch.long), \
               torch.tensor(feature.mem_mask_pos, dtype=torch.long)

    def line2tokens(self, line):
        tokens = line.split("\t")
        return tokens

    def read_example_features(self, data_dir):
        processor = KGProcessor(tokenizer=self.tokenizer, max_mlm_seq_len=self.max_length)
        label_list = processor.get_labels(data_dir)
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        if self.do_train:
            features = processor.get_train_examples(self.e_mask_id, data_dir)
        if self.do_eval:
            features = processor.get_dev_examples(self.e_mask_id, data_dir)
        if self.do_test:
            features = processor.get_test_examples(self.e_mask_id, data_dir)

        return features


class PathqueryDataset(KBCDataset):
    def __init__(self,
                 vocab_path,
                 data_path,
                 max_seq_len=3,
                 vocab_size=-1):
        KBCDataset.__init__(self, vocab_path, data_path, max_seq_len, vocab_size)

    def line2tokens(self, line):
        tokens = []
        s, path, o, mask_type = line.split("\t")
        path_tokens = path.split(",")
        tokens.append(s)
        tokens.extend(path_tokens)
        tokens.append(o)
        tokens.append(mask_type)
        return tokens


if __name__ == '__main__':
    from transformers import BertConfig, BertModel

    # D:/Study/BMKG/kg-bert-master/data/FB15k-237/
    train_dataset = KBCDataset(data_dir='/home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/',
                               do_train=False,
                               do_eval=True,
                               do_test=False,
                               max_seq_len=1024,
                               vocab_size=16396)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
    for iter, batch_data in enumerate(train_loader):
        input_ids, input_mask, seg_ids, pos_ids, mask_label_ids = batch_data
        print(input_ids.shape)
        print(mask_label_ids.shape)
        # print(label.shape)

    # config = BertConfig()
    # config.num_attention_heads = 12
    # config.intermediate_size = 1024
    # config.hidden_size = 768
    # config.max_position_embeddings = 512
    # config.vocab_size = 30522
    bert = BertModel.from_pretrained(
        pretrained_model_name_or_path='/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/'

    )
    config = bert.config
    print(config.hidden_act)
