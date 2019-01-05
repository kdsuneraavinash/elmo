# -*- coding: utf-8 -*-
import collections
import nltk
import os
"""
@author: Sunera
"""

print("# Initializing vocab file creater")


def create_vocab_file(source_directory='train', output_vocab_file='vocab.txt', batch_size=500):
    """
    Creates a vocab file using source directory.
    Batch size determines how much lines to load at once. (Does not affect much)
    """
    unsorted_words, max_word_length = count_words(source_directory, batch_size)

    all_counted = sort(unsorted_words)
    save(all_counted, output_vocab_file)
    print()
    print('Max word length = {}'.format(max_word_length))
    print('Number of tokens = {}'.format(len(all_counted)))
    print("Raw Training prefix = '{}/*'".format(source_directory))
    print("Vocab file = '{}'".format(output_vocab_file))
    print()

    return max_word_length, len(all_counted)


def count_words(directory, batch_size):
    print("* Counting words in {}".format(directory))

    counter = collections.Counter()
    max_word_length = 1

    current_batch_size = 0
    current_batch = 1
    batch_text = ""
    for train_file in os.listdir(directory):
        with open('{}/{}'.format(directory, train_file), 'r', encoding="utf-8", errors='ignore') as f:
            while True:
                line = f.readline().lower()
                if line == "":
                    break  # EOF
                batch_text += line
                current_batch_size += 1
                if current_batch_size == batch_size:
                    max_word_length = max(process_line(
                        batch_text, counter), max_word_length)
                    print(">> Processed {} batch".format(current_batch))

                    current_batch_size = 0
                    current_batch += 1
                    batch_text = ""
    return counter, max_word_length


def process_line(text, counter):
    tokenized = nltk.tokenize.word_tokenize(text)
    counted = collections.Counter(tokenized)
    max_word_length = max(map(len, tokenized))
    counter.update(counted)
    return max_word_length


def sort(all_counted):
    counted_list = [(-all_counted[x], x) for x in all_counted]
    counted_list.sort()
    sorted_vocab = [x[1] for x in counted_list]
    print("* Sorted the word list")
    return sorted_vocab


def save(word_list, output_vocab_file):
    with open(output_vocab_file, 'w', encoding="utf-8") as f:
        f.write('<S>\n</S>\n<UNK>\n')
        for word in word_list:
            f.write(word)
            f.write("\n")

    print("* Vocab file saved in " + output_vocab_file)


if __name__ == '__main__':
    create_vocab_file()
