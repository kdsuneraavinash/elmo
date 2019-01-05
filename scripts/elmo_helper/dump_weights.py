from bilm.training import dump_weights as dw


def save_weights(checkpoint_dir, weights_file):
    dw(checkpoint_dir, weights_file)


if __name__ == '__main__':
    save_weights('checkpoint', 'weights.hdf5')
