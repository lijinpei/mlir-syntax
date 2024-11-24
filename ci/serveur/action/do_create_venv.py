from serveur.action import Action
from serveur.common import get_project_ci_dir
import subprocess
import shlex
import os
import os.path


class CreateVenv(Action):
    def __init__(self):
        pass

    @classmethod
    def desc(cls):
        return "Create a virtual env and fill necessary packages"

    def add_arguments(self, parser):
        parser.add_argument('--path', required=True)
        parser.add_argument('--force', default=False, action='store_true')

    def do_action(self, args):
        path = args.path
        if os.path.exists(path):
            if not args.force:
                return
            else:
                subprocess.run(['rm', '-rf', path], check=True)
        subprocess.run(['python3', '-m', 'venv', path], check=True)
        req_txt = os.path.join(get_project_ci_dir(), 'requirements.txt')
        script = f"""#!/usr/bin/env bash
        set -eo pipefail
        source {path}/bin/activate
        python3 -m pip install -U pip
        python3 -m pip install -r {req_txt}"""
        subprocess.run(['bash', '-c', script], check=True)
