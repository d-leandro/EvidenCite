#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/evidencite-bootstrap.log) 2>&1

dnf install -y docker git jq unzip
command -v aws >/dev/null || dnf install -y awscli-2
systemctl enable --now docker

# AL2023's Docker package may not include the Compose plugin.
install -d -m 0755 /usr/local/lib/docker/cli-plugins
curl -fLsS --retry 3 \
  https://github.com/docker/compose/releases/download/v2.40.0/docker-compose-linux-x86_64 \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
chmod 0755 /usr/local/lib/docker/cli-plugins/docker-compose
/usr/bin/docker compose version

install -d -m 0755 /srv/evidencite/models /srv/evidencite/metrics /srv/evidencite/repo
install -d -m 0700 /srv/evidencite/secrets
# Never write API credentials to user_data or Terraform state.
