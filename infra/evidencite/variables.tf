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
  description = "32 GiB RAM CPU host for Granite INT8 and other EvidenCite models"
  default     = "r6i.xlarge"
}

variable "instance_profile_name" {
  type        = string
  description = "Existing AWS Academy instance profile; verify it can use SSM and read S3"
  default     = "LabInstanceProfile"
}

variable "root_volume_gib" {
  type        = number
  description = "EBS gp3; includes models, images, logs and metrics"
  default     = 100

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
