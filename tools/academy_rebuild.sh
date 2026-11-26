#!/usr/bin/env bash
# Reprovision or redeploy the existing Academy Lite environment. Does not destroy.
# Usage: AWS_PROFILE=evidencite-lab bash tools/academy_rebuild.sh --approve-costs
set -euo pipefail
[[ "${1:-}" == '--approve-costs' ]] || { echo 'Usage: bash tools/academy_rebuild.sh --approve-costs'; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1 AWS_PAGER=''
SHA="$(git rev-parse HEAD)"
DEPLOY_BRANCH="${EVIDENCITE_DEPLOY_BRANCH:-feature/app-infrastructure}"
REMOTE_SHA="$(git ls-remote origin "refs/heads/$DEPLOY_BRANCH" | awk '{print $1}')"
[[ -n "$REMOTE_SHA" && "$SHA" == "$REMOTE_SHA" ]] || {
  echo '[STOP] Push the reviewed commit to $DEPLOY_BRANCH before deployment.' >&2; exit 2;
}
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
STATE_BUCKET="evidencite-tfstate-${ACCOUNT}"
aws s3api head-bucket --bucket "$STATE_BUCKET" >/dev/null
# Retain the EXISTING backend key; renaming a directory must not orphan state.
terraform -chdir=infra/evidencite init -input=false -reconfigure \
  -backend-config="bucket=$STATE_BUCKET" \
  -backend-config='key=evidencite/ec2-demo.tfstate' \
  -backend-config='region=us-east-1' \
  -backend-config='use_lockfile=true'
terraform -chdir=infra/evidencite apply -input=false -auto-approve -var='allow_paid_resources=true'
INSTANCE_ID="$(terraform -chdir=infra/evidencite output -raw instance_id)"
BUCKET="$(terraform -chdir=infra/evidencite output -raw models_bucket)"
# Terraform does not start stopped instances: make this script safe to rerun after a pause.
STATE="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)"
case "$STATE" in
  stopped)
    echo "[EC2] Starting stopped instance: $INSTANCE_ID"
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    ;;
  pending)
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    ;;
  running) ;;
  *) echo "[STOP] Unsupported EC2 state: $STATE" >&2; exit 2 ;;
esac
IP="$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
[[ "$IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || { echo '[STOP] Invalid IP'; exit 2; }
URL="https://evidencite.${IP//./-}.sslip.io"
echo "[EC2] $INSTANCE_ID  [URL] $URL"
# SSM agent may take a while to register after a clean creation.
for attempt in $(seq 1 90); do
  STATUS="$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)"
  [[ "$STATUS" == Online ]] && break
  sleep 10
done
[[ "${STATUS:-}" == Online ]] || { echo '[STOP] SSM instance not online'; exit 1; }
PARAM_FILE="$(mktemp)"
trap 'rm -f "$PARAM_FILE"' EXIT
python3 tools/prepare_ssm_deploy.py "$SHA" "$BUCKET" "$PARAM_FILE"
COMMAND_ID="$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --comment "EvidenCite rebuild $SHA" \
  --parameters "file://$PARAM_FILE" \
  --query Command.CommandId --output text)"
echo "[SSM] Deployment command: $COMMAND_ID"
for attempt in $(seq 1 480); do
  STATUS="$(aws ssm get-command-invocation --command-id "$COMMAND_ID" \
    --instance-id "$INSTANCE_ID" --query Status --output text 2>/dev/null || true)"
  case "$STATUS" in
    Success) echo '[OK] SSM deploy completed'; break ;;
    Failed|Cancelled|TimedOut|Cancelling)
      aws ssm get-command-invocation --command-id "$COMMAND_ID" \
        --instance-id "$INSTANCE_ID" \
        --query '{Status:Status,Output:StandardOutputContent,Errors:StandardErrorContent}'
      exit 1 ;;
  esac
  sleep 15
done
[[ "$STATUS" == Success ]] || { echo '[STOP] SSM deployment timed out'; exit 1; }
for attempt in $(seq 1 40); do
  if curl -fsS --max-time 15 "$URL/health" > /dev/null; then
    echo "[OK] Public HTTPS works: $URL"
    exit 0
  fi
  sleep 10
done
echo "[STOP] Public HTTPS health check failed: $URL/health" >&2
exit 1
