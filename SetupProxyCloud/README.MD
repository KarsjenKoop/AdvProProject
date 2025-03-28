# Squid Proxy on Google Cloud 🚀

This project uses Terraform to provision Squid proxy instances on Google Cloud. It includes firewall rules for inbound traffic on port 3128 and a startup script to install and configure Squid with basic authentication. A Python notebook is also provided to test the deployed proxy instances.

⸻

Table of Contents
*	Requirements
*	Installation & Setup
*	Configuration
*	Deployment
*	Testing the Proxies
*	Project Structure
*	Troubleshooting 🛠️

⸻

# Requirements

## Software
	•	Terraform (version 0.12+ recommended)
	•	Python 3.x (for testing proxies)
	•	Google Cloud SDK (to authenticate and manage GCP resources)

## Python Dependencies

Install the following Python packages (using pip or your favorite package manager):

pip install requests

## Google Cloud
*	An active Google Cloud account with billing enabled.
*	Proper IAM permissions and a valid project.
*	Authentication configured via gcloud auth login or service account credentials.

⸻

Installation & Setup
## 1.	Terraform Initialization:

Ensure you have configured your Google Cloud credentials. Then, run:

```bash
terraform init
```

## 2.	Configure Variables:
Review the variables.tf file. Update the default values as needed for:
*	project
*	region
*	zone
*	instance_count
*	squid_username
*	squid_password

Note: The password is marked as sensitive. Use secure practices when handling credentials.

⸻

# Configuration

Main Terraform File: main.tf

This file sets up:
*	The Google Cloud provider configuration.
*	A firewall rule to allow inbound TCP traffic on port 3128.
*	The compute instances with a startup script that installs and configures Squid.

Example snippet:
```terraform
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
```
The instance resource creates VMs with a startup script that:
	•	Updates the system.
	•	Installs Squid and apache2-utils.
	•	Configures Squid to use basic authentication.
	•	Restarts Squid to apply changes.

⸻

# Deployment
1.	Plan the Deployment:

```
terraform plan
```

2.	Apply the Configuration:
```
terraform apply
```

Confirm the action when prompted. Once completed, Terraform outputs the public IPs of your proxy instances.

⸻

# Testing the Proxies

A Python notebook (testproxies.ipynb) is provided to test the proxies. It performs the following:
*	Runs terraform output -json to retrieve proxy IP addresses.
*	Tests each proxy by sending a GET request to http://httpbin.org/ip using the Squid credentials.

How to Run the Notebook:
1.	Ensure your Squid credentials in the notebook match those in variables.tf.
2.	Launch Jupyter Notebook:

jupyter notebook testproxies.ipynb


3.	Execute the Notebook Cells to see test results for each proxy.

⸻

Project Structure
```
├── main.tf              # Terraform configuration for GCP resources
├── variables.tf         # Variable definitions for Terraform
└── testproxies.ipynb    # Python notebook to test the deployed proxies
```


⸻

Troubleshooting 🛠️
*	Terraform Errors:
Verify your Google Cloud credentials and check your IAM permissions. Use terraform plan to debug configuration issues.
*	Proxy Issues:
Ensure the firewall rule allows traffic on port 3128. Confirm that the Squid startup script executed correctly by checking the instance logs via the Google Cloud Console.
*	Python Test Failures:
Double-check that the credentials in the Python notebook match the ones provided in variables.tf. Increase the timeout if necessary.
