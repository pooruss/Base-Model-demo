import logging
import argparse
from .args import ArgumentGroup


def create_args():
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
    model_g.add_arg("model_name", str, "coke", "Model name")
    model_g.add_arg("hidden_size", int, 256, "CoKE model config: hidden size, default 256")
    model_g.add_arg("num_hidden_layers", int, 12, "CoKE model config: num_hidden_layers, default 12")
    model_g.add_arg("num_attention_heads", int, 4, "CoKE model config: num_attention_heads, default 4")
    model_g.add_arg("vocab_size", int, 16396, "CoKE model config: vocab_size")
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
    train_g.add_arg("use_cuda", bool, False, "Use cuda or not.")
    train_g.add_arg("gpus", int, 4, "GPU nums.")
    train_g.add_arg("node", int, 1, "Node nums.")
    train_g.add_arg("epoch", int, 400, "Number of epoches for training.")
    train_g.add_arg("warm_up_epochs", int, 40, "Number of epoches for training.")
    train_g.add_arg("checkpoint_num", int, 100, "Checkpoint num.")
    train_g.add_arg("learning_rate", float, 5e-4, "Learning rate used to train with warmup.")
    train_g.add_arg("warmup_proportion", float, 0.1,
                    "Proportion of training steps to perform linear learning rate warmup for.")
    train_g.add_arg("weight_decay", float, 0.01, "Weight decay rate for L2 regularizer.")
    train_g.add_arg("soft_label", float, 0.8, "Value of soft labels for loss computation")
    train_g.add_arg("save_path", str, './checkpoints1/', "Use cuda or not.")

    log_g = ArgumentGroup(parser, "logging", "logging related.")
    log_g.add_arg("skip_steps", int, 30, "The steps interval to print loss.")

    data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
    data_g.add_arg("vocab_path", str, './data/FB15K/vocab.txt', "Vocab path.")
    data_g.add_arg("train_data_path", str, './data/FB15K/train.coke.txt', "Train data for coke.")
    data_g.add_arg("valid_data_path", str, './data/FB15K/valid.coke.txt', "Valid data for coke.")
    data_g.add_arg("true_triple_path", str, './data/FB15K/all.txt', "Valid data for coke.")
    data_g.add_arg("max_seq_len", int, 3, "Number of tokens of the longest sequence.")
    data_g.add_arg("batch_size", int, 2048, "Total examples' number in batch for training. see also --in_tokens.")

    args = parser.parse_args()
    return args, logger
