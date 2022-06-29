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
from transformers import AutoModel, AutoTokenizer, AutoConfig, RobertaModel, RobertaTokenizer, RobertaConfig
from tqdm import tqdm
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

torch.manual_seed(1)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
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
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, all_ids, input_mask, segment_ids, position_ids, cls_label):
        self.all_ids = all_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.position_ids = position_ids
        self.cls_label = cls_label

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
            return lines


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

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
        return self._read_tsv(os.path.join(data_dir, "train_ori.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r', encoding='utf-8') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]

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

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):

            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":

                label = "1"

                guid = "%s-%s" % (set_type, i)
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, text_a=head_ent_text, text_b=relation_text, text_c=tail_ent_text, label=label))

            elif set_type == "train":
                guid = "%s-%s" % (set_type, i)
                examples.append(
                    InputExample(guid=guid, text_a=head_ent_text, text_b=relation_text, text_c=tail_ent_text, label="1"))

                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:
                    # corrupting head
                    for j in range(2):
                        tmp_head = ''
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_head_text = ent2text[tmp_head]
                        examples.append(
                            InputExample(guid=guid, text_a=tmp_head_text, text_b=relation_text, text_c=tail_ent_text, label="0"))
                else:
                    # corrupting tail
                    tmp_tail = ''
                    for j in range(2):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[tmp_tail]
                        examples.append(
                            InputExample(guid=guid, text_a=head_ent_text, text_b=relation_text, text_c=tmp_tail_text, label="0"))
        return examples


def _truncate_pad_seq_triple(tokens_a, tokens_b, tokens_c, max_length, max_a_len, max_b_len, max_c_len):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    if len(tokens_a) > max_a_len:
        tokens_a = tokens_a[:max_a_len]
    else:
        tokens_a.extend((max_a_len - len(tokens_a) + 1) * ["[PAD]"])
    if len(tokens_b) > max_b_len:
        tokens_b = tokens_b[:max_b_len]
    else:
        tokens_b.extend((max_b_len - len(tokens_b) + 1) * ["[PAD]"])
    if len(tokens_c) > max_c_len:
        tokens_c = tokens_c[:max_c_len]
    else:
        tokens_c.extend((max_c_len - len(tokens_c) + 1) * ["[PAD]"])

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()
    return tokens_a, tokens_b, tokens_c


class KBCDataset(Dataset):
    """ DataReader
    """

    def __init__(self, data_dir, do_train, do_eval, do_test, max_seq_len=3, vocab_size=-1):
        # D:/Study/BMKG/git_clone/BMKG/bmkg/base_model/
        self.tokenizer = AutoTokenizer.from_pretrained("/home/wanghuadong/liangshihao/KEPLER-huggingface/bert-base/")
        self.vocab = self.tokenizer.get_vocab()
        self.sep_id = self.tokenizer.convert_tokens_to_ids("[SEP")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        self.mask_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        # if vocab_size > 0:
        #     assert len(self.vocab) == vocab_size, \
        #         "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
        #         (vocab_size, len(self.vocab))
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.max_seq_len = max_seq_len

        self.max_head_seq_len = 384
        self.max_rel_seq_len = 8
        self.max_tail_seq_len = 384
        self.max_length = 776

        self.examples = self.read_example(data_dir)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return torch.tensor(example.all_ids, dtype=torch.long), \
               torch.tensor(example.input_mask, dtype=torch.long), \
               torch.tensor(example.segment_ids, dtype=torch.long), \
               torch.tensor(example.position_ids, dtype=torch.long), \
               torch.tensor(example.cls_label, dtype=torch.long)

    def line2tokens(self, line):
        tokens = line.split("\t")
        return tokens

    def read_example(self, data_dir):
        processor = KGProcessor()
        label_list = processor.get_labels(data_dir)
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        if self.do_train:
            examples = processor.get_train_examples(data_dir)
        if self.do_eval:
            examples = processor.get_dev_examples(data_dir)
        if self.do_test:
            examples = processor.get_test_examples(data_dir)
        for ex_index, example in tqdm(enumerate(examples)):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            head_tokens = self.tokenizer.tokenize(example.text_a)
            rel_tokens = self.tokenizer.tokenize(example.text_b)
            tail_tokens = self.tokenizer.tokenize(example.text_c)
            cls_label = label_map[example.label]

            head_tokens, rel_tokens, tail_tokens = _truncate_pad_seq_triple(head_tokens, rel_tokens, tail_tokens, self.max_length - 4,
                                     self.max_head_seq_len - 2, self.max_rel_seq_len - 1, self.max_tail_seq_len - 1)

            all_tokens = ["[CLS]"] + head_tokens + ["[SEP]"]
            head_tokens = []
            segment_ids = [0] * len(all_tokens)
            input_mask = []
            pos_ids = []
            for i in range(len(all_tokens)):
                pos_ids += [i]
                input_mask += [0 if all_tokens[i] == "[PAD]" else 1]

            rel_tokens += ["[SEP]"]
            all_tokens += rel_tokens
            segment_ids += [1] * (len(rel_tokens))
            for i in range(len(rel_tokens)):
                pos_ids += [i]
                input_mask += [0 if rel_tokens[i] == "[PAD]" else 1]
            rel_tokens = []

            tail_tokens += ["[SEP]"]
            all_tokens += tail_tokens
            segment_ids += [0] * (len(tail_tokens))
            for i in range(len(tail_tokens)):
                pos_ids += [i]
                input_mask += [0 if tail_tokens[i] == "[PAD]" else 1]
            tail_tokens = []

            all_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
            all_tokens = []

            features.append(
                InputFeatures(all_ids=all_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              position_ids=pos_ids,
                              cls_label=cls_label))
        # np.save('processed_input_data.npy', np.array(features))

        # features = np.load('processed_input_data.npy', allow_pickle=True)
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