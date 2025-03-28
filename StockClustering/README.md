# Stock Autoencoder Clustering

Stock Autoencoder Clustering is a Python-based pipeline for processing stock data, building a convolutional autoencoder to extract latent features from time series, clustering stocks using various methods, and visualizing the results interactively.

## Features

- **Data Preprocessing:**  
  Loads stock data from a CSV file while preserving specific string values (e.g., `'NA'`), converts dates, sorts data, and computes normalized stock indices.

- **Autoencoder Model:**  
  Constructs a 1D convolutional autoencoder to learn latent representations of stock time series.

- **Clustering:**  
  Supports multiple clustering techniques, including:
  - KMeans
  - Agglomerative Clustering
  - DBSCAN
  - Gaussian Mixture Models

- **Visualization:**  
  Generates interactive line plots for each cluster using Plotly Express.

- **Model Persistence:**  
  Provides methods to save the trained autoencoder and encoder models to disk and to load them back for further use.

## Installation

### 1. 

```bash
# Create the virtual environment
python -m venv stockclusteringenv

# Activate the virtual environment:
# On Windows:
stockclusteringenv\Scripts\activate
# On macOS/Linux:
source stockclusteringenv/bin/activate

# Install Jupyter and ipykernel inside the virtual environment
pip install -r requirements.txt

# Create a new Jupyter kernel for this environment
python -m ipykernel install --user --name stockclusteringenv --display-name "stockclusteringenv"
```