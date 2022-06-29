""" data reader for CoKE
"""
from torch.utils.data import Dataset

import six
import collections
import logging
import torch
from .mask import prepare_batch_data

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

RawExample = collections.namedtuple("RawExample", ["token_ids", "mask_type"])


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


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


def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    output = []
    for item in tokens:
        output.append(vocab[item])
    return output


class KBCDataset(Dataset):
    """ DataReader
    """

    def __init__(self, vocab_path, data_path, max_seq_len=3, vocab_size=-1):
        self.vocab = load_vocab(vocab_path)
        if vocab_size > 0:
            assert len(self.vocab) == vocab_size, \
                "Assert Error! Input vocab_size(%d) is not consistant with voab_file(%d)" % \
                (vocab_size, len(self.vocab))
        self.pad_id = self.vocab["[PAD]"]
        self.mask_id = self.vocab["[MASK]"]
        self.max_seq_len = max_seq_len
        self.examples = self.read_example(data_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        token_ids, mask_type = example[0], example[1]
        example_out = [token_ids] + [mask_type]
        example_data = prepare_batch_data(
            [example_out],
            max_len=self.max_seq_len,
            pad_id=self.pad_id,
            mask_id=self.mask_id)
        src_id, pos_id, input_mask, mask_pos, mask_label = example_data[0], example_data[1], example_data[2], \
                                                           example_data[3], example_data[4]

        return torch.tensor(src_id, dtype=torch.long), \
               torch.tensor(pos_id, dtype=torch.long), \
               torch.tensor(input_mask, dtype=torch.float16), \
               torch.tensor(mask_pos, dtype=torch.long), \
               torch.tensor(mask_label, dtype=torch.long)

    def line2tokens(self, line):
        tokens = line.split("\t")
        return tokens

    def read_example(self, input_file):
        """Reads the input file into a list of examples."""
        examples = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                line = convert_to_unicode(line.strip())
                tokens = self.line2tokens(line)
                assert len(tokens) <= (self.max_seq_len + 1), \
                    "Expecting at most [max_seq_len + 1]=%d tokens each line, current tokens %d" \
                    % (self.max_seq_len + 1, len(tokens))
                token_ids = convert_tokens_to_ids(self.vocab, tokens[:-1])
                if len(token_ids) <= 0:
                    continue
                examples.append(
                    RawExample(
                        token_ids=token_ids, mask_type=tokens[-1]))
        return examples


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
