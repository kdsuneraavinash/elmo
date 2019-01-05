# -*- coding: utf-8 -*-
import random
import nltk
import os
"""
@author: Sunera
"""

print("# Initializing Preprocessor")


def preprocess(directory, save_directory, notify_multiplier=100):
    print("WARNING: This won't randomize your data.")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_n = 0
    for file in os.listdir(directory):
        f_read_file = open('{}/{}'.format(directory, file),
                           'r', encoding="utf-8", errors='ignore')
        f_write_file = open('{}/{}'.format(save_directory, file),
                            'w', encoding="utf-8", errors='ignore')

        text = f_read_file.read().lower()

        tokenized = nltk.tokenize.sent_tokenize(text)
        tokenized_words = (nltk.tokenize.casual.casual_tokenize(l) for l in tokenized)
        tokenized = [" ".join(l) for l in tokenized_words]
        
        random.shuffle(tokenized)
        for sent in tokenized:
            f_write_file.write(sent.replace("\n", " ") + '\n')

        file_n += 1
        if file_n % notify_multiplier == 0:
            print("Processed {} files".format(file_n))

        f_read_file.close()
        f_write_file.close()

    print("* Testing files preprocesed")
    print("* Testing prefix = '{}/*'".format(save_directory))


if __name__ == '__main__':
    preprocess('train', 'train_f')
