"""
data preprocess for CoKE KBC datasets
"""
import os
import collections
import argparse


def write_vocab(output_file, entity_lst, relation_lst):
    fout = open(output_file, "w")
    fout.write("[PAD]" + "\n")
    for i in range(95):
        fout.write("[unused{}]\n".format(i))
    fout.write("[UNK]" + "\n")
    fout.write("[CLS]" + "\n")
    fout.write("[SEP]" + "\n")
    fout.write("[MASK]" + "\n")
    for e in entity_lst.keys():
        fout.write(e + "\n")
    for r in relation_lst.keys():
        fout.write(r + "\n")
    vocab_size = 100 + len(entity_lst) + len(relation_lst)
    print(">> vocab_size: %s" % vocab_size)
    fout.close()


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


def get_unique_entities_relations(train_file, dev_file, test_file):
    entity_lst = dict()
    relation_lst = dict()
    all_files = [train_file, dev_file, test_file]
    for input_file in all_files:
        print("dealing %s" % train_file)
        with open(input_file, "r") as f:
            for line in f.readlines():
                tokens = line.strip().split("\t")
                assert len(tokens) == 3
                entity_lst[tokens[0]] = len(entity_lst)
                entity_lst[tokens[2]] = len(entity_lst)
                relation_lst[tokens[1]] = len(relation_lst)
    print(">> Number of unique entities: %s" % len(entity_lst))
    print(">> Number of unique relations: %s" % len(relation_lst))
    return entity_lst, relation_lst


def write_true_triples(train_file, dev_file, test_file, vocab, output_file):
    true_triples = []
    all_files = [train_file, dev_file, test_file]
    for input_file in all_files:
        with open(input_file, "r") as f:
            for line in f.readlines():
                h, r, t = line.strip('\r \n').split('\t')
                assert (h in vocab) and (r in vocab) and (t in vocab)
                hpos = vocab[h]
                rpos = vocab[r]
                tpos = vocab[t]
                true_triples.append((hpos, rpos, tpos))

    print(">> Number of true triples: %d" % len(true_triples))
    fout = open(output_file, "w")
    for hpos, rpos, tpos in true_triples:
        fout.write(str(hpos) + "\t" + str(rpos) + "\t" + str(tpos) + "\n")
    fout.close()


def generate_mask_type(input_file, output_file):
    with open(output_file, "w") as fw:
        with open(input_file, "r") as fr:
            for line in fr.readlines():
                fw.write(line.strip('\r \n') + "\tMASK_HEAD\n")
                fw.write(line.strip('\r \n') + "\tMASK_TAIL\n")


def kbc_data_preprocess(train_file, dev_file, test_file, vocab_path,
                        true_triple_path, new_train_file, new_dev_file,
                        new_test_file):
    entity_lst, relation_lst = get_unique_entities_relations(
        train_file, dev_file, test_file)
    write_vocab(vocab_path, entity_lst, relation_lst)
    vocab = load_vocab(vocab_path)
    write_true_triples(train_file, dev_file, test_file, vocab,
                       true_triple_path)

    generate_mask_type(train_file, new_train_file)
    generate_mask_type(dev_file, new_dev_file)
    generate_mask_type(test_file, new_test_file)


def get_args():
    parser = argparse.ArgumentParser()

    # for mask data
    parser.add_argument("--mask_task", type=str, required=True, default=None,
                        help="mask task name: fb15k, fb15k237, wn18rr, wn18, pathqueryFB, pathqueryWN"
                        )
    parser.add_argument("--dir", type=str, required=True, default=None, help="task data directory")
    parser.add_argument("--mask_train", type=str, required=False, default="mask_train.txt",
                        help="train file name, default train.txt")
    parser.add_argument("--mask_valid", type=str, required=False, default="mask_valid.txt",
                        help="valid file name, default valid.txt")
    parser.add_argument(
        "--test", type=str, required=False, default="test.txt", help="test file name, default test.txt")

    # for text data
    parser.add_argument("--text", type=str, help="path to original text file")
    parser.add_argument("--train", type=str, help="path to original training data file")
    parser.add_argument("--valid", type=str, help="path to original validation data file")
    parser.add_argument("--converted_text", type=str, default="Qdesc.txt", help="path to converted text file")
    parser.add_argument("--converted_train", type=str, default="train.txt", help="path to converted training file")
    parser.add_argument("--converted_valid", type=str, default="valid.txt", help="path to converted validation file")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    task = args.task.lower()
    assert task in ["fb15k", "wn18", "fb15k237", "wn18rr", "wikidata5m"]
    raw_train_file = os.path.join(args.dir, args.train)
    raw_dev_file = os.path.join(args.dir, args.valid)
    raw_test_file = os.path.join(args.dir, args.test)

    vocab_file = os.path.join(args.dir, "vocab.txt")
    true_triple_file = os.path.join(args.dir, "all.txt")
    new_train_file = os.path.join(args.dir, "train.coke.txt")
    new_test_file = os.path.join(args.dir, "test.coke.txt")
    new_dev_file = os.path.join(args.dir, "valid.coke.txt")

    kbc_data_preprocess(raw_train_file, raw_dev_file, raw_test_file,
                        vocab_file, true_triple_file, new_train_file,
                        new_dev_file, new_test_file)
