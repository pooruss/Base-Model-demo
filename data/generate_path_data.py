import sys
import csv
from tqdm import tqdm

def build_path_data(input_file, output_file, quotechar=None):
    """Reads a tab separated value file."""
    hr_set = set()
    hr_t_dict = dict()
    h_rt_dict = dict()
    fout = open(output_file, "w", encoding='utf-8')

    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in tqdm(reader):
            line = list(cell for cell in line)
            head, rel, tail = line
            hr_t_dict[head + '\t' + rel] = tail
            h_rt_dict[head] = rel + '\t' + tail
            lines.append(line)
        print(len(h_rt_dict))
        cnt = 0
        for line in lines:
            head, rel, tail = line
            if tail in h_rt_dict:
                cnt += 1
                fout.write(head + '\t' + rel + '\t' + tail + '\t' + h_rt_dict[tail] + '\n')
                print(head + '\t' + rel + '\t' + tail + '\t' + h_rt_dict[tail])
        print(cnt)
        return lines


build_path_data(sys.argv[1], sys.argv[2])