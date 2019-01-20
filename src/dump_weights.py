import argparse
from bilm.training import dump_weights as dw


def main(args):
    dw(args.save_dir, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--out', help='Output file location')

    args = parser.parse_args()
    main(args)
