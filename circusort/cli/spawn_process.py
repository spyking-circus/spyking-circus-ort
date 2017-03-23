import argparse

from circusort.cli.process import main



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host')
    parser.add_argument('-a', '--address')
    parser.add_argument('-l', '--log-address')

    args = parser.parse_args()
    args = vars(args)

    main(args)
