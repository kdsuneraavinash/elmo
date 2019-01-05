from bilm.training import test, load_options_latest_checkpoint, load_vocab
from bilm.data import LMDataset, BidirectionalLMDataset


def run_test(test_files='test_f/*',
             checkpoint_dir='checkpoint',
             vocab_file='vocab.txt',
             batch_size=100):

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
    run_test()
