import argparse
from serveur.os import all_os
from serveur.action import Action
from serveur.common import get_project_ci_dir
from datetime import datetime
import subprocess
import os


class BuildDockerImage(Action):
    @classmethod
    def desc(cls):
        return "Build docker image"

    @classmethod
    def construct_arg_for_docker(cls, arg_dict):
        res = []
        for k, v in arg_dict.items():
            res.append('--build-arg')
            res.append(f'{k}={v}')
        return res

    def prepare_distro_args(self, distro):
        if distro == 'debian':
            return self.construct_arg_for_docker({'APT_SOURCE_TUNA': 'on'})
        return []

    def prepare_proxy_args(self, proxy):
        if proxy is None:
            return []
        return self.construct_arg_for_docker({
            'http_proxy': proxy,
            'https_proxy': proxy,
            'HTTP_PROXY': proxy,
            'HTTPS_PROXY': proxy,
        })

    def add_arguments(self, parser):
        parser.add_argument('--distro', required=True,
                            choices=[os.name for os in all_os])
        parser.add_argument('--registry', required=True)
        parser.add_argument('--llvm', default=20)
        parser.add_argument('--tag', default=None)

    def do_action(self, args):
        ci_dir = get_project_ci_dir()
        env_dir = os.path.join(ci_dir, 'environment')
        distro_dir = os.path.join(env_dir, args.distro)
        tag = args.tag
        if tag is None:
            tag = datetime.utcnow().strftime('%Y_%m_%d.%H_%M_%S')
        image_name = f"{args.registry}/mlir-syntax-{args.distro}:{tag}"
        proxy_args = self.prepare_proxy_args(args.proxy)
        distro_args = self.prepare_distro_args(args.distro)
        common_args = self.construct_arg_for_docker({'LLVM_VER': args.llvm})
        subprocess.run(['docker', 'build', '-t', image_name, '--no-cache', '--network',
                       'host', *proxy_args, *distro_args, '.'], cwd=distro_dir, check=True)
        subprocess.run(['docker', 'push', image_name],
                       cwd=distro_dir, check=True)
        if tag != 'latest':
            latest_image_name = f"{args.registry}/mlir-syntax-{args.distro}:latest"
            subprocess.run(['docker', 'tag', image_name, latest_image_name],
                           cwd=distro_dir, check=True)
            subprocess.run(['docker', 'push', latest_image_name],
                           cwd=distro_dir, check=True)

        subprocess.run(['docker', 'rmi', image_name],
                       cwd=distro_dir, check=True)
