import argparse
import sys

from circusort.cli.process import main



if __name__ == '__main__':

    #sys.stdout.write("spawn process...\n")
    #sys.stdout.flush()  # required

    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host')
    parser.add_argument('-a', '--address')
    parser.add_argument('-l', '--log-address')

    args = parser.parse_args()
    args = vars(args)

    main(args)

    #sys.stdout.write("process spawned\n")
    #sys.stdout.flush()  # required

    sys.exit(0)
