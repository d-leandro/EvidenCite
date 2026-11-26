# EvidenCite Academy Lite: reproducible deployment

## Persistent preconditions (created once; not destroyed by this Terraform)

- AWS Academy credentials: local AWS CLI profile `evidencite-lab`.
- Private S3 bucket `evidencite-demo-<account-id>-models`, containing the four verified
  model directories (`tfidf_A`, `tfidf_B`, `scibert_A`, `scibert_B`). The bucket **and its
  block-public-access and encryption configuration** must be managed independently of
  this ephemeral Terraform stack; a destroy must never remove their protections.
- SecureString `/evidencite/prod/gemini-api-key` in **us-east-1**, retrievable by the
  existing `LabInstanceProfile` role. Never put secrets in Terraform state, Git or SSM command text.
- S3 remote backend bucket `evidencite-tfstate-<account-id>` kept separately.
- Repository branch `feature/app-infrastructure` pushed to GitHub.

## Deploy a fresh instance or refresh a running one

```bash
export AWS_PROFILE=evidencite-lab AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
bash tools/academy_rebuild.sh --approve-costs
```

The script runs `terraform apply`, waits for SSM, checks out the pushed Git SHA,
retrieves Gemini from Parameter Store **inside EC2**, syncs the four models from S3,
builds containers, derives the current public IP through IMDSv2 and configures Caddy HTTPS.
`/health` must succeed with valid TLS. Public URL follows the current IP, for example
`https://evidencite.54-175-68-140.sslip.io`.

**Important:** changing IP after a stop/start requires re-running this deployment.
`terraform plan` showing `No changes` does *not* update the domain in the instance.
Do not leave the instance running after a demo; EC2 compute consumes lab budget.

## Safe destruction

Export important metrics from `/srv/evidencite/metrics/` **before** destroy, if needed.
Root EBS is configured with `delete_on_termination=true`; these files cannot be
recovered after destroy without an external backup.

```bash
terraform -chdir=infra/evidencite destroy -var='allow_paid_resources=true'
```

The S3 model bucket, its security configuration, the S3 backend bucket and the
SSM SecureString survive `destroy`. Do **not** run `destroy` until the one-time state
migration below is complete. After migration, confirm a normal Terraform plan does
**not** contain S3 security-resource deletions.

### One-time migration: detach persistent bucket protections from ephemeral state

The settings are already configured on the model bucket. `terraform state rm` removes
only Terraform's record; it **does not** delete anything on AWS. Run once from the
repository root, after this code change and before any Terraform plan/apply/destroy:

```bash
terraform -chdir=infra/evidencite state rm \
  aws_s3_bucket_public_access_block.models \
  aws_s3_bucket_server_side_encryption_configuration.models

BUCKET="$(terraform -chdir=infra/evidencite output -raw models_bucket)"
aws s3api get-public-access-block --bucket "$BUCKET"
aws s3api get-bucket-encryption --bucket "$BUCKET"
terraform -chdir=infra/evidencite plan -var='allow_paid_resources=true'
```

Expected: all four public-access-block booleans remain `true`, encryption `AES256`,
and Terraform shows no S3 resource destroy actions. The settings and bucket must not
be removed manually. Do not recreate these resources in this ephemeral stack.

## Limits and security

- The `gemini-3.5-flash-lite` inference endpoint is reached from the API using the
  encrypted Parameter Store value. `max_retries=2` mitigates transient HTTP 503.
- Nginx rate limits `/api/v1/predict` to 6 requests/minute per IP plus a burst of 6.
  **This is not a financial spending cap**: configure API quotas/budgets at Google too.
- Granite is intentionally not deployed to this 8 GiB instance.
- GitHub Actions uses temporary AWS Academy credentials and may require refreshing
  repository secrets when the Academy session changes.
- Caddy HTTP/HTTPS is public; SSH and direct API ports are not.
