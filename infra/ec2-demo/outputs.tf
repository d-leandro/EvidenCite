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

output "public_ip" {
  value       = aws_instance.demo.public_ip
  description = "Ephemeral EC2 public IPv4; changes after stop/start"
}

output "public_url" {
  value       = "https://evidencite.${replace(aws_instance.demo.public_ip, ".", "-")}.sslip.io"
  description = "Demo URL. Run the deployment script after any IP change."
}
