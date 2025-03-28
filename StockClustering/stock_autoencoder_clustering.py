import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, UpSampling1D, Dense, Flatten, Reshape, Cropping1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import plotly.express as px

class StockAutoencoderClustering:
    def __init__(self, csv_file, timesteps=251, latent_dim=8):
        """
        Initializes the clustering pipeline.

        Parameters:
            csv_file (str): Path to the CSV file.
            timesteps (int): Number of time steps to use (default is 251).
            latent_dim (int): Dimension of the latent space.
        """
        self.csv_file = csv_file
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.df = None
        self.df_pivot = None
        self.data = None
        self.data_normalized = None
        self.autoencoder = None
        self.encoder = None
        self.latent_vectors = None
        self.cluster_results = {}

        self._load_data()

    def _load_data(self):
        """Loads the CSV, sets data types, computes normalized index, and sorts the DataFrame."""
        # Read CSV with 'NA' preserved as string
        df = pd.read_csv(self.csv_file, keep_default_na=False)
        df['Ticker'] = df['Ticker'].astype(str)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Ticker', 'Date'])
        # Normalize the 'Adj Close' price for each ticker
        df['Index'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x / x.iloc[0] * 100)
        self.df = df

    def prepare_data(self):
        """
        Pivots the DataFrame so that each ticker's normalized index series is a column,
        fills missing values, ensures a fixed number of timesteps, and reshapes the data.
        """
        df_pivot = self.df.pivot(index='Date', columns='Ticker', values='Index')
        # Fill missing values using forward and backward fill
        df_pivot = df_pivot.fillna(method='ffill').fillna(method='bfill')
        # Ensure exactly 'timesteps' rows (e.g., first 251 dates)
        df_pivot = df_pivot.iloc[:self.timesteps]
        self.df_pivot = df_pivot
        # Transpose so that each row corresponds to one stock and add channel dimension
        data = df_pivot.T.values  # shape: (num_stocks, timesteps)
        data = data[..., np.newaxis]  # new shape: (num_stocks, timesteps, 1)
        self.data = data

    def normalize_data(self):
        """
        Standardizes each stock's time series to have a mean of 0 and standard deviation of 1.
        """
        if self.data is None:
            raise ValueError("Data not prepared. Run prepare_data() first.")
        data = self.data
        data_normalized = np.empty_like(data)
        for i in range(data.shape[0]):
            scaler = StandardScaler()
            data_normalized[i, :, 0] = scaler.fit_transform(data[i, :, 0].reshape(-1, 1)).flatten()
        self.data_normalized = data_normalized

    def build_autoencoder(self):
        """
        Builds and compiles the 1D convolutional autoencoder.
        Note: The architecture is chosen so that after encoding the flattened dimension is 4032,
        which is then reshaped (63, 64) in the decoder.
        """
        input_length = self.timesteps
        input_dim = 1
        latent_dim = self.latent_dim

        input_layer = Input(shape=(input_length, input_dim))
        # Encoder
        x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(input_layer)         # (251, 16)
        x = Conv1D(16, kernel_size=3, activation='relu', strides=2, padding='same')(x)            # (126, 16)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)                       # (126, 32)
        x = Flatten()(x)                                                                        # (4032,)
        x = Dense(256, activation='relu')(x)
        latent = Dense(latent_dim, activation='relu')(x)

        # Decoder
        x = Dense(256, activation='relu')(latent)
        x = Dense(4032, activation='relu')(x)
        x = Reshape((63, 64))(x)
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)                       # (63, 64)
        x = UpSampling1D(2)(x)                                                                    # (126, 64)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)                       # (126, 32)
        x = UpSampling1D(2)(x)                                                                    # (252, 32)
        x = Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)                       # (252, 16)
        x = Conv1D(1, kernel_size=3, activation='linear', padding='same')(x)                      # (252, 1)
        output_layer = Cropping1D(cropping=(0, 1))(x)                                             # Crop to (251, 1)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer=Adam(), loss='mse')
        self.autoencoder = autoencoder
        # Build encoder model for latent vector extraction
        self.encoder = Model(inputs=input_layer, outputs=latent)

    def train_autoencoder(self, epochs=1000, batch_size=64):
        """
        Trains the autoencoder on the normalized data.

        Parameters:
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size.
        Returns:
            History object from model.fit().
        """
        if self.data_normalized is None:
            raise ValueError("Data not normalized. Run normalize_data() first.")
        history = self.autoencoder.fit(self.data_normalized, self.data_normalized, epochs=epochs, batch_size=batch_size)
        return history

    def extract_latent_vectors(self):
        """
        Uses the encoder to extract latent vectors from the (un-normalized) data.

        Returns:
            NumPy array of latent vectors.
        """
        if self.encoder is None or self.data is None:
            raise ValueError("Ensure that the autoencoder is built and data is prepared.")
        self.latent_vectors = self.encoder.predict(self.data)
        return self.latent_vectors

    def cluster_data(self, n_clusters=20, method='kmeans'):
        """
        Clusters the stocks based on their latent vectors using the specified method.

        Parameters:
            n_clusters (int): Number of clusters (used for methods that require it).
            method (str): Clustering method; one of 'kmeans', 'agglomerative', 'dbscan', 'gmm'.

        Returns:
            Cluster labels as a NumPy array.
        """
        if self.latent_vectors is None:
            self.extract_latent_vectors()

        method = method.lower()
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = model.fit_predict(self.latent_vectors)
        elif method == 'agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)
            clusters = model.fit_predict(self.latent_vectors)
        elif method == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
            clusters = model.fit_predict(self.latent_vectors)
        elif method == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            clusters = model.fit_predict(self.latent_vectors)
        else:
            raise ValueError("Unknown clustering method: choose from 'kmeans', 'agglomerative', 'dbscan', or 'gmm'.")

        self.cluster_results[method] = clusters
        return clusters

    def cluster_all_methods(self, n_clusters=10):
        """
        Applies multiple clustering methods and stores the results.

        Parameters:
            n_clusters (int): Number of clusters for methods that require a set number.
        
        Returns:
            Dictionary with clustering method names as keys and their cluster labels as values.
        """
        if self.latent_vectors is None:
            self.extract_latent_vectors()

        results = {}
        results['kmeans'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(self.latent_vectors)
        results['agglomerative'] = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(self.latent_vectors)
        results['dbscan'] = DBSCAN(eps=0.5, min_samples=5).fit_predict(self.latent_vectors)
        results['gmm'] = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(self.latent_vectors)
        self.cluster_results = results
        return results

    def get_cluster_dataframe(self, method='kmeans'):
        """
        Combines ticker names with their cluster labels into a DataFrame.

        Parameters:
            method (str): Clustering method whose results to return.

        Returns:
            DataFrame with columns 'Ticker' and 'Cluster'.
        """
        if self.df_pivot is None:
            raise ValueError("Data not prepared. Run prepare_data() first.")
        tickers = self.df_pivot.columns
        clusters = self.cluster_results.get(method.lower(), None)
        if clusters is None:
            raise ValueError(f"No clustering result for method '{method}'. Run cluster_data() first.")
        return pd.DataFrame({'Ticker': tickers, 'Cluster': clusters})

    def plot_clusters(self, method='kmeans'):
        """
        Plots interactive line charts for each cluster using Plotly Express.

        Parameters:
            method (str): Clustering method to visualize.
        """
        cluster_df = self.get_cluster_dataframe(method)
        df_plot = self.df_pivot.reset_index()  # Reset index so that 'Date' becomes a column

        # Loop through each unique cluster and plot the time series for stocks in that cluster.
        for cluster in sorted(cluster_df['Cluster'].unique()):
            tickers_in_cluster = cluster_df[cluster_df['Cluster'] == cluster]['Ticker'].tolist()
            if not tickers_in_cluster:
                continue
            fig = px.line(
                df_plot,
                x='Date',
                y=tickers_in_cluster,
                title=f"{method.capitalize()} - Cluster {cluster}",
                labels={'value': 'Normalized Index', 'Date': 'Date'}
            )
            fig.update_layout(legend_title_text='Ticker')
            fig.show()

    def save_models(self, base_path):
        """
        Saves the autoencoder and encoder models to disk.

        Parameters:
            base_path (str): Base file path (without extension) to save the models.
                             Two files will be created: '<base_path>_autoencoder.keras' and '<base_path>_encoder.keras'.
        """
        if self.autoencoder is None or self.encoder is None:
            raise ValueError("Models have not been built yet. Build or train the model first.")
        self.autoencoder.save(f"{base_path}_autoencoder.keras")
        self.encoder.save(f"{base_path}_encoder.keras")
        print(f"Models saved to '{base_path}_autoencoder.keras' and '{base_path}_encoder.keras'.")

    def load_models(self, base_path):
        """
        Loads the autoencoder and encoder models from disk.

        Parameters:
            base_path (str): Base file path (without extension) where the models are saved.
                             It will look for '<base_path>_autoencoder.keras' and '<base_path>_encoder.keras'.
        """
        self.autoencoder = tf.keras.models.load_model(f"{base_path}_autoencoder.keras")
        self.encoder = tf.keras.models.load_model(f"{base_path}_encoder.keras")
        print(f"Models loaded from '{base_path}_autoencoder.keras' and '{base_path}_encoder.keras'.")