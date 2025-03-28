variable "project" {
  description = "Google Cloud project ID"
}

variable "region" {
  description = "Google Cloud region"
  default     = "eu-west4"
}

variable "zone" {
  description = "Google Cloud zone"
  default     = "eu-west4-a"
}

variable "instance_count" {
  description = "Number of proxy instances to create"
  default     = 8
}

variable "squid_username" {
  description = "The username for Squid proxy authentication"
  type        = string
  default     = "username"
}

variable "squid_password" {
  description = "The password for Squid proxy authentication"
  type        = string
  sensitive = true
  default     = "password"
}