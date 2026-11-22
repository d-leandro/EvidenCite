output "instance_id" {
  value       = aws_instance.demo.id
  description = "Use with SSM Session Manager and SSM port forwarding"
}

output "models_bucket" {
  value       = local.models_bucket_name
  description = "Private bucket for model artifacts"
}

output "region" {
  value = var.aws_region
}

output "ssh_required" {
  value = false
}
