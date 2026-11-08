output "api_ecr_url" {
  value = aws_ecr_repository.api.repository_url
}

output "app_ecr_url" {
  value = aws_ecr_repository.app.repository_url
}

output "alb_dns_name" {
  value = aws_lb.evidencite.dns_name
}