import argparse
from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset


def main(args):
    checkpoint_dir = args.save_dir
    vocab_file = args.vocab_file
    test_files = args.test_prefix
    batch_size = args.batch_size

    options, ckpt_file = load_options_latest_checkpoint(checkpoint_dir)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    vocab = load_vocab(vocab_file, max_word_length)

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    if options.get('bidirectional'):
        data = BidirectionalLMDataset(test_files, vocab, **kwargs)
    else:
        data = LMDataset(test_files, vocab, **kwargs)

    test(options, ckpt_file, data, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--test_prefix', help='Prefix for test files')
    parser.add_argument('--batch_size', help='Batch size for each GPU', type=int, default=32)

    args = parser.parse_args()
    main(args)
