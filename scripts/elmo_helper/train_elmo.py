from bilm.training import train, load_vocab
from bilm.data import BidirectionalLMDataset
import os


def train_model(max_word_length, n_train_tokens,
                batch_size=16,
                n_gpus=1,
                train_files='train_f/*',
                vocab_file='vocab.txt',
                checkpoint_dir='checkpoint'):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # load the vocab
    vocab = load_vocab(vocab_file, max_word_length=max_word_length)

    # define the options
    options = {
        'bidirectional': True,

        'char_cnn': {'activation': 'relu',
                     'embedding': {'dim': 16},
                     'filters': [[1, 32],
                                 [2, 32],
                                 [3, 64],
                                 [4, 128],
                                 [5, 256],
                                 [6, 512],
                                 [7, 1024]],
                     'max_characters_per_token': max_word_length,
                     'n_characters': 261,
                     'n_highway': 2},

        'dropout': 0.1,

        'lstm': {
            'cell_clip': 3,
            'dim': 4096,
            'n_layers': 2,
            'proj_clip': 3,
            'projection_dim': 512,
            'use_skip_connections': True},

        'all_clip_norm_val': 10.0,

        'n_epochs': 10,
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,
        'n_negative_samples_batch': 8192,
    }

    prefix = train_files
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)

    tf_save_dir = checkpoint_dir
    tf_log_dir = checkpoint_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    train_model(1695, 14)
