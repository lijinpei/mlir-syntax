import sys
from serveur.common import get_common_arg_parser_args
import importlib
import os
import os.path
import argparse


def create_main_arg_parser():
    parser = argparse.ArgumentParser(
        prog='serveur', description='Helper program to build this project and pack up external depencies', **get_common_arg_parser_args())
    parser.add_argument('--proxy', default=None)
    return parser


def register_actions(arg_parser):
    pkg_dir = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'action')
    res = {}
    sub_parsers = arg_parser.add_subparsers(dest='action')
    for file in os.listdir(pkg_dir):
        if not file.startswith('do_') or not file.endswith('.py'):
            continue
        if not os.path.isfile(os.path.join(pkg_dir, file)):
            continue
        mod_name = file.removesuffix('.py')
        action_name = mod_name.removeprefix('do_')
        mod = importlib.import_module('.' + mod_name, package='serveur.action')
        action_class_name = "".join(x.capitalize()
                                    for x in action_name.split("_"))
        action = getattr(mod, action_class_name)()
        res[action_name] = action
        sub_parser = sub_parsers.add_parser(
            action_name, prog=action_class_name, description=action.desc(), **get_common_arg_parser_args())
        action.add_arguments(sub_parser)
    return res


def export_proxy(proxy):
    if proxy is None:
        return
    for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        os.environ[k] = proxy


def main():
    arg_parser = create_main_arg_parser()
    actions = register_actions(arg_parser)
    args = arg_parser.parse_args(sys.argv[1:])
    export_proxy(args.proxy)
    actions[args.action].do_action(args)
