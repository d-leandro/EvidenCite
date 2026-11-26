variable "aws_region" {
  type        = string
  description = "AWS Academy lab region"
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Resource prefix"
  default     = "evidencite-demo"
}

variable "instance_type" {
  type        = string
  description = "x86_64 instance with >= 8 GiB RAM; Academy Lite without Granite before use"
  default     = "t3.large"
}

variable "instance_profile_name" {
  type        = string
  description = "Existing AWS Academy instance profile; verify it can use SSM and read S3"
  default     = "LabInstanceProfile"
}

variable "root_volume_gib" {
  type        = number
  description = "EBS gp3; includes models, images, logs and metrics"
  default     = 60

  validation {
    condition     = var.root_volume_gib >= 40 && var.root_volume_gib <= 100
    error_message = "Use 40-100 GiB; verify the permitted EBS size in AWS Academy."
  }
}

variable "allow_paid_resources" {
  type        = bool
  description = "Must be explicitly true when approving an AWS EC2 creation"
  default     = false
}
