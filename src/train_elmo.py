import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

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
      'max_characters_per_token': 50,
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
    
     'n_epochs': args.n_epochs,
     'n_train_tokens': args.n_train_tokens,
     'batch_size': args.batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    train(options, data, args.n_gpus, args.save_dir, args.log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--log_dir', help='Location of log files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    
    parser.add_argument('--train_prefix', help='Prefix for train files')

    parser.add_argument('--batch_size', help='Batch size for each GPU', type=int, default=32)
    parser.add_argument('--n_gpus', help='Number of GPUs', type=int, default=1)

    parser.add_argument('--n_train_tokens', help='Number of tokens in training data', type=int)

    parser.add_argument('--n_epochs', help='Number of epochs', type=int, default=10)

    args = parser.parse_args()
    main(args)
