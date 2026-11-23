#!/usr/bin/env python3
"""Build JSON parameters for AWS SSM without putting API secrets in command text."""
import json
import re
import shlex
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        raise SystemExit('Uso: prepare_ssm_deploy.py <sha40> <bucket> <output.json>')
    sha, bucket, out = sys.argv[1:]
    if not re.fullmatch(r'[0-9a-fA-F]{40}', sha):
        raise SystemExit('Invalid commit SHA')
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]{2,62}', bucket):
        raise SystemExit('Invalid bucket')

    repo = '/srv/evidencite/repo'
    commands = (
        'set -euo pipefail; cloud-init status --wait >/dev/null; '
        f'if [ ! -d {repo}/.git ]; then '
        f'git clone https://github.com/d-leandro/EvidenCite.git {repo}; '
        'fi; '
        f'git -C {repo} fetch origin {sha}; '
        f'git -C {repo} checkout --detach {sha}; '
        f'EVIDENCITE_MODELS_BUCKET={shlex.quote(bucket)} '
        f'bash {repo}/tools/ec2_academy_deploy.sh'
    )
    payload = {'commands': ['sudo bash -lc ' + shlex.quote(commands)], 'executionTimeout': ['7200']}
    Path(out).write_text(json.dumps(payload), encoding='utf-8')
    print('SSM parameters saved, no API credentials included.')


if __name__ == '__main__':
    main()
