import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


class PCAReconstructionAnomalyDetector:
    """
    PCA reconstruction baseline for ECG anomaly detection.

    The model is usually fitted on normal beats. Beats with high
    reconstruction error are treated as anomalies.
    """

    def __init__(self, n_components=8, threshold_percentile=95, random_state=42):
        self.n_components = n_components
        self.threshold_percentile = threshold_percentile
        self.model = PCA(n_components=n_components, random_state=random_state)
        self.threshold = None
    
    def fit(self, X_train):
        """Fit PCA and set the anomaly threshold from training errors."""
        X_train = self._flatten_if_needed(X_train)
        self.model.fit(X_train)

        train_scores = self.anomaly_score(X_train)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)

        return self
    
    def predict(self, X_test):
        """Predict anomalies (0 = normal, 1 = anomaly)."""
        scores = self.anomaly_score(X_test)
        return (scores > self.threshold).astype(int)
    
    def anomaly_score(self, X_test):
        """Return reconstruction error for each beat."""
        X_test = self._flatten_if_needed(X_test)
        encoded = self.model.transform(X_test)
        reconstructed = self.model.inverse_transform(encoded)
        return np.mean((X_test - reconstructed) ** 2, axis=1)

    def _flatten_if_needed(self, X):
        X = np.asarray(X)
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X


class ARMAModel:
    """
    ARMA (AutoRegressive Moving Average) model for anomaly detection.

    Each beat is treated as a short time series. The anomaly score is the
    reconstruction/prediction error of an ARIMA(p, 0, q) model fitted to
    that beat.
    """
    
    def __init__(self, p=1, d=0, q=1, threshold_percentile=95):
        self.p = p
        self.d = d
        self.q = q
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        
    def fit(self, X_train):
        """Set the anomaly threshold from training beat errors."""
        train_scores = self.anomaly_score(X_train)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)
        
        return self
    
    def predict(self, X_test):
        """Predict anomalies (0 = normal, 1 = anomaly)."""
        scores = self.anomaly_score(X_test)
        return (scores > self.threshold).astype(int)

    def anomaly_score(self, X):
        """Return one ARMA reconstruction error for each beat."""
        beats = self._prepare_beats(X)
        return np.array([self._score_one_beat(beat) for beat in beats])

    def _score_one_beat(self, beat):
        beat = np.asarray(beat, dtype=float)

        try:
            model = ARIMA(beat, order=(self.p, self.d, self.q))
            result = model.fit()
            fitted = np.asarray(result.fittedvalues)

            if len(fitted) != len(beat):
                fitted = fitted[-len(beat):]

            return np.mean((beat[-len(fitted):] - fitted) ** 2)
        except Exception:
            return np.inf

    def _prepare_beats(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            return X.reshape(1, -1)
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X


class LSTMAutoencoderTorch(nn.Module):
    """
    Small PyTorch LSTM autoencoder for ECG beat reconstruction.

    Input shape: (batch, sequence_length, 1)
    Output shape: (batch, sequence_length, 1)
    """

    def __init__(self, seq_len=180, hidden_size=32, latent_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.to_latent(hidden[-1])
        decoder_start = self.from_latent(latent)
        decoder_input = decoder_start.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder_lstm(decoder_input)
        return self.output_layer(decoded)


class LSTMAutoencoderAnomalyDetector:
    """
    Reconstruction-based anomaly detector using a PyTorch LSTM autoencoder.

    The detector is fitted on normal beats. The anomaly score is the mean
    squared reconstruction error for each beat.
    """

    def __init__(
        self,
        seq_len=180,
        hidden_size=32,
        latent_dim=8,
        threshold_percentile=95,
        learning_rate=1e-3,
        batch_size=32,
        epochs=20,
        device=None,
        random_state=42,
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMAutoencoderTorch(seq_len, hidden_size, latent_dim).to(self.device)
        self.threshold = None
        self.history = []

    def fit(self, X_train):
        """Train the autoencoder and set the anomaly threshold."""
        torch.manual_seed(self.random_state)

        X_train = self._prepare_tensor(X_train)
        dataset = TensorDataset(X_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model.train()
        self.history = []

        for _ in range(self.epochs):
            epoch_losses = []

            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                loss = loss_fn(reconstructed, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            self.history.append(float(np.mean(epoch_losses)))

        train_scores = self.anomaly_score(X_train)
        self.threshold = np.percentile(train_scores, self.threshold_percentile)

        return self

    def predict(self, X_test):
        """Predict anomalies (0 = normal, 1 = anomaly)."""
        scores = self.anomaly_score(X_test)
        return (scores > self.threshold).astype(int)

    def anomaly_score(self, X):
        """Return reconstruction error for each beat."""
        X_tensor = self._prepare_tensor(X).to(self.device)

        self.model.eval()
        scores = []

        with torch.no_grad():
            for start in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[start:start + self.batch_size]
                reconstructed = self.model(batch)
                batch_scores = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
                scores.extend(batch_scores.cpu().numpy())

        return np.array(scores)

    def _prepare_tensor(self, X):
        if isinstance(X, torch.Tensor):
            X_tensor = X.detach().float().cpu()
        else:
            X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)

        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(-1)
        if len(X_tensor.shape) != 3:
            raise ValueError("Expected X with shape (n_beats, seq_len) or (n_beats, seq_len, 1).")

        return X_tensor


class VAE(keras.Model):
    """
    Variational Autoencoder for ECG anomaly detection.
    """
    
    def __init__(self, input_shape, latent_dim=8):
        super(VAE, self).__init__()
        self.input_shape_val = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv1D(64, 3, strides=2, padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(16, activation='relu'),
        ])
        
        # Latent space
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(16, activation='relu'),
            layers.Reshape((4, 4)),
            layers.Conv1DTranspose(64, 3, strides=2, padding='same', activation='relu'),
            layers.Conv1DTranspose(32, 3, strides=2, padding='same', activation='relu'),
            layers.Conv1D(1, 3, padding='same', activation='sigmoid'),
            layers.Flatten(),
        ])
    
    def encode(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed
    
    def vae_loss(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        
        # Reconstruction loss
        recon_loss = tf.reduce_mean(tf.square(x - reconstructed))
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        
        return recon_loss + kl_loss


class LiquidNeuralNetwork(keras.Model):  #Placeholder for Liquid Neural Network implementation
    """
    Liquid Neural Network (LNN) for ECG anomaly detection.
    Uses continuous-time neural networks with liquid dynamics.
    
    Reference: Hasani et al., "Liquid Time-Constant Networks"
    """
    
    def __init__(self, input_shape, hidden_size=32):
        super(LiquidNeuralNetwork, self).__init__()
        self.input_shape_val = input_shape
        self.hidden_size = hidden_size
        
        # Simplified LNN implementation
        self.input_layer = layers.Input(shape=input_shape)
        self.dense1 = layers.Dense(hidden_size, activation='tanh')
        self.dense2 = layers.Dense(hidden_size, activation='tanh')
        self.output_layer = layers.Dense(input_shape[1])
    
    def call(self, x):
        h = self.dense1(x)
        h = self.dense2(h)
        return self.output_layer(h)
