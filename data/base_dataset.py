""" data reader for CoKE
"""
from torch.utils.data import Dataset
from transformers import BertTokenizer
from .preprocess import *

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

torch.manual_seed(1)


class BaseDataset(Dataset):
    """ DataReader
    """
    def __init__(self, data_dir, do_train, do_eval, do_test, max_seq_len=512):
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
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.max_seq_len = max_seq_len
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

    def read_example_features(self, data_dir):
        processor = KGProcessor(tokenizer=self.tokenizer, max_mlm_seq_len=self.max_seq_len)
        label_list = processor.get_labels(data_dir)
        # label_map = {label: i for i, label in enumerate(label_list)}
        features = []
        if self.do_train:
            features = processor.get_train_examples(self.e_mask_id, data_dir)
        if self.do_eval:
            features = processor.get_dev_examples(self.e_mask_id, data_dir)
        if self.do_test:
            features = processor.get_test_examples(self.e_mask_id, data_dir)

        return features


if __name__ == '__main__':
    from transformers import BertConfig, BertModel

    # D:/Study/BMKG/kg-bert-master/data/FB15k-237/
    train_dataset = BaseDataset(data_dir='/home/wanghuadong/liangshihao/kg-bert-master/data/FB15k-237/',
                                do_train=False,
                                do_eval=True,
                                do_test=False,
                                max_seq_len=512)
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
