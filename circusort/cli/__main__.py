import argparse


parser = argparse.ArgumentParser(description='Command Line Interface')

subparsers = parser.add_subparsers()
parser_manager = subparsers.add_parser('manager', help='manager')
subparser_manager = parser_manager.add_subparsers()
parser_manager_create = subparser_manager.add_parser('create', help='create')
parser_manager_create.add_argument('-a', '--address', help='specify the IP address/hostname')
parser_manager_create.add_argument('-p', '--port', help='specify the port number')

args = parser.parse_args()
args = vars(args)

print(args)
