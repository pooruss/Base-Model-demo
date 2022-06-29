import math
from abc import ABC
from typing import Any, Generator, Tuple
import collections
import numpy as np
import random

from .._data import TripleDataBatch
from .mask import prepare_batch_data


class MaskTripleDataBatch(TripleDataBatch):
    src_ids: np.array
    pos_id: np.array
    input_mask: np.array
    mask_pos: np.array
    mask_label: np.array

    def __init__(self,
                 src_ids: np.array,
                 pos_id: np.array,
                 input_mask: np.array,
                 mask_pos: np.array,
                 mask_label: np.array) -> None:
        super().__init__(src_ids[:0], src_ids[:1], src_ids[:2])
        src_ids = TripleDataBatch(src_ids[:0], src_ids[:1], src_ids[:2])


class TripleDataset:
    """
    Dataset is responsible for reading given data file and yield DataBatch.

    TripleDataset yields TripleDataBatch from a specific range of a given .npy file.
    """

    def __init__(self, filename: str, start: int = 0, end: int = -1, batch_size: int = 20, shuffle: bool = False):
        super(TripleDataset).__init__()
        assert start >= 0
        self.data = np.load(filename)
        if end == -1:
            end = self.data.shape[0]
        assert start < end
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Generator[TripleDataBatch, Any, None]:
        iter_start = self.start
        iter_end = self.end

        def iterator():
            starts = list(range(iter_start, iter_end, self.batch_size))
            if self.shuffle:
                random.shuffle(starts)
            while True:
                for cur in starts:
                    batch = self.data[cur: cur + self.batch_size]
                    data = TripleDataBatch(batch[:, 0], batch[:, 1], batch[:, 2])
                    yield data

        return iterator()

    def __len__(self):
        return math.ceil((self.end - self.start) / self.batch_size)


class MaskTripleDataset:
    """
    Dataset is responsible for reading given data file and yield DataBatch.

    MaskTripleDataset yields MaskedTripleDataBatch from a specific range of a given .npy file.
    """

    def __init__(self,
                 filename: str,
                 vocab_path: str,
                 start: int = 0,
                 end: int = -1,
                 batch_size: int = 20,
                 shuffle: bool = False,
                 max_seq_len: int = 3):
        super(MaskTripleDataset).__init__()
        assert start >= 0
        assert start < end
        self.data = np.load(filename)
        if end == -1:
            end = self.data.shape[0]
        assert start < end
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start = start
        self.end = end
        self.max_seq_len = max_seq_len
        self.pad_id = self.pad_id(self.load_vocab(vocab_path))
        self.mask_id = self.mask_id(self.load_vocab(vocab_path))

    def __iter__(self) -> Generator[MaskTripleDataBatch, Any, None]:
        iter_start = self.start
        iter_end = self.end

        def iterator():
            starts = list(range(iter_start, iter_end, self.batch_size))
            if self.shuffle:
                random.shuffle(starts)
            while True:
                for cur in starts:
                    batch = self.data[cur: cur + self.batch_size]
                    batch = self.prepare_data(batch)
                    data = MaskTripleDataBatch(batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3], batch[:, 4])
                    yield data

        return iterator()

    def __len__(self):
        return math.ceil((self.end - self.start) / self.batch_size)

    def prepare_data(self, worker_data):
        masked_data = []
        for mask_triple in worker_data:
            token_ids, mask_type = [mask_triple[0], mask_triple[1], mask_triple[2]], mask_triple[3]
            example_out = [token_ids] + [mask_type]
            example_data = prepare_batch_data([example_out], self.max_seq_len, self.pad_id, self.mask_id)
            masked_data.append([example_data[0],   # src_id
                                example_data[1],   # pos_id
                                example_data[2],   # input_mask
                                example_data[3],   # mask_pos
                                example_data[4]])  # mask_label
        return np.array(masked_data)

    def load_vocab(self, vocab_file):
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

    def pad_id(self, vocab):
        """
        :param vocab: vocab for certain dataset
        :return: the id of [PAD] in vocab
        """
        return vocab["PAD"]

    def mask_id(self, vocab):
        """
        :param vocab: vocab for certain dataset
        :return: the id of [MASK] in vocab
        """
        return vocab["MASK"]
