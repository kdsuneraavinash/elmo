import argparse
import os
import re
import collections
import sys


def main():
    word_counter = collections.Counter()
    vocab = args.vocab
    src = args.src
    min_occurrences = args.min_occurrences
    max_letters = args.max_letters
    n_load_files = args.load_files

    if src is None:
        raise SystemExit("src cannot be null")

    loaded_text = []
    for i, file in enumerate(os.listdir(src)):
        file_path = src + file
        if os.path.isfile(file_path):
            with open(file_path) as f:
                loaded_text.append(f.read())
        if (i + 1) % n_load_files == 0:
            tmp_counted = count_words(loaded_text)
            word_counter.update(tmp_counted)
            loaded_text = []
    if loaded_text:
        tmp_counted = count_words(loaded_text)
        word_counter.update(tmp_counted)

    print("Original vocab size =", len(word_counter))
    remove_min_occurrences(word_counter, min_occurrences)
    print("Vocab size after min occurrences removed =", len(word_counter))
    remove_max_length(word_counter, max_letters)
    print("Vocab size after long words removed =", len(word_counter))
    remove_special_characters(word_counter)
    print("Vocab size after special characters removed =", len(word_counter))
    print("Sorting words...")
    sorted_keys = sort(word_counter)
    with open(vocab + ".log", "w") as fw:
        for word in sorted_keys:
            fw.write(("{:>" + str(max_letters) + "} = {}\r\n").format(word, word_counter[word]))
    with open(vocab, "w") as fw:
        fw.write("<S>\n</S>\n<UNK>\n")
        for word in sorted_keys:
            fw.write(word + "\n")
    print(word_counter)


def count_words(loaded_file_text):
    n_all_files = len(loaded_file_text)

    split_by_line = []
    for i, file_text in enumerate(loaded_file_text):
        for word in re.split("[\n ]", file_text.strip()):
            word = word.strip()
            if word == "":
                continue
            split_by_line.append(word)
        sys.stdout.write("Processed {}/{} files \r".format(i + 1, n_all_files))
        sys.stdout.flush()
    print()
    return collections.Counter(split_by_line)


def remove_min_occurrences(counted, thresh):
    print("Removing occurrences less than {}".format(thresh))
    keys = list(counted.keys())
    n_all_keys = len(keys)
    for i, key in enumerate(keys):
        if counted[key] < thresh:
            del counted[key]
        sys.stdout.write("Words formatted: {}/{} \r".format(i + 1, n_all_keys))
        sys.stdout.flush()
    print()
    return counted


def remove_max_length(counted, thresh):
    print("Removing words longer than {}".format(thresh))
    keys = list(counted.keys())
    n_all_keys = len(keys)
    for i, key in enumerate(keys):
        if len(key) > thresh:
            del counted[key]
        sys.stdout.write("Words formatted: {}/{} \r".format(i + 1, n_all_keys))
        sys.stdout.flush()
    print()
    return counted


def remove_special_characters(counted):
    print("Removing special characters")
    keys = list(counted.keys())
    n_all_keys = len(keys)
    for i, key in enumerate(keys):
        tmp_key = re.sub(r"[a-zA-Z0-9#_><&=+\"\'*/\\{}()$\[\]]", "", key)
        if len(tmp_key) == 0:
            del counted[key]
        elif tmp_key != key:
            counted[tmp_key] = counted.get(tmp_key, 0) + counted[key]
            del counted[key]
        sys.stdout.write("Words formatted: {}/{} \r".format(i + 1, n_all_keys))
        sys.stdout.flush()
    print()
    return counted


def sort(counted):
    counted_keys = list(counted)
    counted_keys.sort(key=counted.get, reverse=True)
    return counted_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", help="Output files path",
                        default="vocab.txt")
    parser.add_argument("--src", help="Input files name")
    parser.add_argument("--load_files", help="Input number of files to load at a time", type=int, default=15)
    parser.add_argument("--min_occurrences",
                        help="Minimum occurrences", type=int, default=3)
    parser.add_argument(
        "--max_letters", help="Maximum Characters occurrences", type=int, default=25)
    args = parser.parse_args()
    main()
