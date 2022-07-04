""" data reader for CoKE
"""
from torch.utils.data import Dataset
import collections
import logging


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = line.strip().split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


def load_desc(desc_file):
    desc_dict = dict()
    for line in open(desc_file, 'r', encoding='utf-8'):
        try:
            ent, desc = line.strip().split('\t')
        except:
            u = line.strip().split('\t')
            ent, desc = u[0], u[1:]
            desc = ' '.join(desc)
        desc_dict[ent] = desc
    return desc_dict
    # with open(desc_file, "r") as f:
    #     for line in f.readlines():
    #         # u = line.strip().split('\t')
    #         try:
    #             ent, desc = line.strip().split('\t')
    #         except:
    #             u = line.strip().split('\t')
    #             print(len(u))
    #             continue
    #         desc_dict[ent] = desc
    # return desc_dict


def convert_tokens_to_descs(vocab, desc_dict, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    output = []
    for item in tokens:
        try:
            output.append(desc_dict[item])
        except:
            output.append(vocab[item])
    if len(output) != 3:
        print(output)
    return output


class DescDataset(Dataset):
    """ DataReader
    """

    def __init__(self, vocab_path, desc_path, data_path, max_seq_len=3, vocab_size=-1):
        self.vocab = load_vocab(vocab_path)
        if vocab_size > 0:
            assert len(self.vocab) == vocab_size, \
                "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
                (vocab_size, len(self.vocab))
        self.pad_id = self.vocab["[PAD]"]
        self.mask_id = self.vocab["[MASK]"]
        self.max_seq_len = max_seq_len
        self.desc_dict = load_desc(desc_path)
        self.examples = self.read_example(data_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        h_desc, r_desc, t_desc = example[0], example[1], example[2]
        return h_desc, r_desc, t_desc

    def read_example(self, input_file):
        """Reads the input file into a list of examples."""
        examples = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split("\t")
                token_descs = convert_tokens_to_descs(self.vocab, self.desc_dict, tokens)
                if len(token_descs) <= 0:
                    continue
                examples.append(token_descs)
        return examples
