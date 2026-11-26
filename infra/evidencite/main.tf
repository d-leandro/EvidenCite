data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ami" "amazon_linux_2023" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-2023.*-x86_64"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Public HTTP/HTTPS terminate at Caddy; SSH and direct API ports stay closed.
resource "aws_security_group" "demo" {
  name_prefix = "${var.project_name}-"
  description = "EvidenCite demo: no public ingress"
  vpc_id      = data.aws_vpc.default.id

  # Only expose the TLS-terminating Caddy proxy; SSH and 8080 remain closed.
  ingress {
    description = "HTTP for ACME challenge and HTTPS redirect"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS EvidenCite evaluator access"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Outbound for SSM, S3, packages and APIs"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Project = "EvidenCite" }
}

locals {
  models_bucket_name = "${var.project_name}-${data.aws_caller_identity.current.account_id}-models"
}
# The model bucket and its access-block/encryption settings are PERSISTENT.
# They are intentionally outside this ephemeral Terraform stack so that a
# destroy/apply cycle never deletes the model artifacts or their protections.
# One-time migration: detach the two existing resources from Terraform state
# BEFORE running plan/apply/destroy; instructions in REBUILD.md.

resource "aws_instance" "demo" {
  ami                         = data.aws_ami.amazon_linux_2023.id
  instance_type               = var.instance_type
  subnet_id                   = sort(data.aws_subnets.default.ids)[0]
  vpc_security_group_ids      = [aws_security_group.demo.id]
  associate_public_ip_address = true # dynamic public IP for Caddy HTTPS
  iam_instance_profile        = var.instance_profile_name
  user_data                   = file("${path.module}/bootstrap.sh")

  root_block_device {
    volume_size           = var.root_volume_gib
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true # BACK UP metrics/model artifacts before destroy
  }

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  credit_specification {
    cpu_credits = "standard" # prevent T3 Unlimited surplus CPU-credit charges; throttling possible
  }

  lifecycle {
    # Preserve the running instance when bootstrap.sh changes.
    # New instances still receive the latest bootstrap.
    ignore_changes = [user_data]

    precondition {
      condition     = var.allow_paid_resources
      error_message = "Deployment blocked: set -var='allow_paid_resources=true' only after checking lab limits and costs."
    }
  }

  tags = {
    Name    = "${var.project_name}-ec2"
    Project = "EvidenCite"
  }
}
