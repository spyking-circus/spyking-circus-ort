import argparse
import sys

from circusort.manager import Manager



def manager_parser():
    parser = argparse.ArgumentParser(description='Launch a manager.')
    parser.add_argument('-a', '--address', help='specify the IP address/hostname')
    parser.add_argument('-p', '--port', help='specify the port number')
    return parser

def main():
    parser = manager_parser()
    args = parser.parse_args()
    args = vars(args)
    print(args)
    # create_manager(**args)
    # manager = Manager(**args)
    # manager.set(address=args.address)
    # manager.set(port=args.port)


if __name__ == '__main__':
    main()
