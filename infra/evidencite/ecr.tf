resource "aws_ecr_repository" "api" {
  name         = "${var.project_name}-api"
  force_delete = true
}

resource "aws_ecr_repository" "app" {
  name         = "${var.project_name}-app"
  force_delete = true
}