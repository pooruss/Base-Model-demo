import argparse
import logging

from .._preprocess import preprocess


def main():
    FORMAT = '%(levelname)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--data_files', type=str, required=True, nargs='+')
    parser.add_argument('--data_format', choices=['hrt', 'htr', 'rht', 'rth', 'thr', 'trh'], default='hrt')
    parser.add_argument('--union_vocab', action='store_true', default=False, help="Whether to use same embedding "
                                                                                  "space for entities and relations.")
    parser.add_argument('--output_path', type=str, default="[DEFAULT]", help="The output path. Defaults to a directory"
                                                                             "in the `data` dir named with the last"
                                                                             "path of the input data path")
    parser.add_argument('--partition', type=int, default=1, help="Whether to run graph partition algorithm "
                                                                 "and number of parts.")

    args = parser.parse_args()
    preprocess(args)


if __name__ == '__main__':
    main()
