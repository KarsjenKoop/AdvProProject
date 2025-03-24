provider "google" {
  project = var.project
  region  = var.region
}

# Firewall rule to allow inbound traffic on port 3128 for Squid
resource "google_compute_firewall" "squid_proxy" {
  name    = "allow-squid-proxy"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["3128"]
  }
  
  direction     = "INGRESS"
  source_ranges = ["0.0.0.0/0"]  # Adjust this to restrict access as needed
  target_tags   = ["squid-proxy"]
}

resource "google_compute_instance" "proxy" {
  count        = var.instance_count
  name         = "proxy-${count.index}"
  machine_type = "f1-micro"
  zone         = var.zone

  tags = ["squid-proxy"]

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update -y
    # Install Squid and apache2-utils for creating a password file.
    apt-get install -y squid apache2-utils

    # Create a password file using the provided credentials from variables.
    htpasswd -b -c /etc/squid/passwd ${var.squid_username} ${var.squid_password}

    # Backup the original Squid configuration.
    cp /etc/squid/squid.conf /etc/squid/squid.conf.bak

    # Write a new Squid configuration that requires authentication.
    cat << 'EOC' > /etc/squid/squid.conf
    # Squid configuration with basic authentication enabled
    auth_param basic program /usr/lib/squid/basic_ncsa_auth /etc/squid/passwd
    auth_param basic children 5
    auth_param basic realm Squid Proxy
    auth_param basic credentialsttl 2 hours

    acl authenticated proxy_auth REQUIRED

    # Allow authenticated users.
    http_access allow authenticated

    # Define the port Squid listens on.
    http_port 3128

    # Optionally, allow local network access.
    acl localnet src 10.0.0.0/8
    http_access allow localnet

    # Deny all other access.
    http_access deny all

    # Set a visible hostname.
    visible_hostname proxy-instance
    EOC

    # Restart Squid to apply changes.
    systemctl restart squid
  EOF
}

output "proxy_ips" {
  value = google_compute_instance.proxy[*].network_interface[0].access_config[0].nat_ip
}