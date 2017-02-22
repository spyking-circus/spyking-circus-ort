import argparse

from .manager import main



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--interface', required=True)
    parser.add_argument('-p', '--port', required=True)
    parser.add_argument('-l', '--log_address', required=True)

    args = parser.parse_args()
    args = vars(args)

    main(args)
