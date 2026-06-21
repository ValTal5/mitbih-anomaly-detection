import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ncps.torch import CfC
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

    The encoder is an LSTM that compresses the beat into a latent vector. The
    decoder is a dense (fully-connected) head that reconstructs the whole beat
    from that latent vector. A recurrent decoder fed a repeated constant tends to
    collapse to a flat output, so a dense decoder is used instead (same approach
    as the VAE). Beats are normalized to [0, 1], hence the final sigmoid.

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
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, seq_len),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _, (hidden, _) = self.encoder_lstm(x)
        latent = self.to_latent(hidden[-1])
        decoded = self.decoder(latent)
        return decoded.unsqueeze(-1)


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


class VAETorch(nn.Module):
    """
    Small fully-connected Variational Autoencoder for ECG beats.

    Input shape:  (batch, input_dim)
    Output shape: (batch, input_dim)

    The encoder maps a beat to a Gaussian latent distribution
    (z_mean, z_log_var). The decoder reconstructs the beat from a sample of
    that distribution. Beats are assumed to be normalized to [0, 1], so the
    decoder ends with a sigmoid.
    """

    def __init__(self, input_dim=180, hidden_size=64, latent_dim=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.to_mean = nn.Linear(hidden_size, latent_dim)
        self.to_log_var = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mean(h), self.to_log_var(h)

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        return reconstructed, z_mean, z_log_var


class VAEAnomalyDetector:
    """
    Reconstruction-based anomaly detector using a PyTorch VAE.

    The VAE is fitted on normal beats by maximizing the evidence lower bound
    (reconstruction term + beta * KL term). The anomaly score is the mean
    squared reconstruction error, computed from the latent mean (no sampling)
    for a stable, reproducible score. Using reconstruction error keeps this
    detector comparable with the PCA and LSTM-autoencoder baselines.
    """

    def __init__(
        self,
        input_dim=180,
        hidden_size=64,
        latent_dim=8,
        beta=1.0,
        threshold_percentile=95,
        learning_rate=1e-3,
        batch_size=32,
        epochs=20,
        device=None,
        random_state=42,
    ):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.threshold_percentile = threshold_percentile
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAETorch(input_dim, hidden_size, latent_dim).to(self.device)
        self.threshold = None
        self.history = []

    def fit(self, X_train):
        """Train the VAE on normal beats and set the anomaly threshold."""
        torch.manual_seed(self.random_state)

        X_train = self._prepare_tensor(X_train)
        dataset = TensorDataset(X_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        self.history = []

        for _ in range(self.epochs):
            epoch_losses = []

            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstructed, z_mean, z_log_var = self.model(batch)

                # ELBO = reconstruction loss + beta * KL divergence (summed over
                # features, averaged over the batch).
                recon_loss = torch.mean(torch.sum((batch - reconstructed) ** 2, dim=1))
                kl_loss = -0.5 * torch.mean(
                    torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), dim=1)
                )
                loss = recon_loss + self.beta * kl_loss

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
        """Return reconstruction error for each beat (latent mean, no sampling)."""
        X_tensor = self._prepare_tensor(X).to(self.device)

        self.model.eval()
        scores = []

        with torch.no_grad():
            for start in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[start:start + self.batch_size]
                z_mean, _ = self.model.encode(batch)
                reconstructed = self.model.decode(z_mean)
                batch_scores = torch.mean((batch - reconstructed) ** 2, dim=1)
                scores.extend(batch_scores.cpu().numpy())

        return np.array(scores)

    def _prepare_tensor(self, X):
        if isinstance(X, torch.Tensor):
            X_tensor = X.detach().float().cpu()
        else:
            X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)

        if len(X_tensor.shape) == 3 and X_tensor.shape[-1] == 1:
            X_tensor = X_tensor.squeeze(-1)
        if len(X_tensor.shape) != 2:
            raise ValueError("Expected X with shape (n_beats, input_dim) or (n_beats, input_dim, 1).")

        return X_tensor


class CfCAutoencoderTorch(nn.Module):
    """
    Liquid-network autoencoder for ECG beats, built on CfC cells.

    CfC (Closed-form Continuous-time) cells are the closed-form variant of the
    Liquid Time-Constant networks of Hasani et al. They keep the continuous-time
    "liquid" dynamics (input-dependent time constants) but run as fast as a
    standard RNN.

    The CfC cell is used as the recurrent encoder; the decoder is a dense head
    that reconstructs the whole beat from the latent vector (same design as the
    LSTM autoencoder and the VAE). This keeps the LSTM-vs-CfC comparison on the
    encoder cell, while avoiding the flat-output collapse of a recurrent decoder
    fed a repeated constant. Beats are normalized to [0, 1], hence the sigmoid.

    Input shape:  (batch, sequence_length, 1)
    Output shape: (batch, sequence_length, 1)
    """

    def __init__(self, seq_len=180, hidden_size=32, latent_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.encoder = CfC(input_size=1, units=hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, seq_len),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded, _ = self.encoder(x)
        summary = encoded[:, -1, :]
        latent = self.to_latent(summary)
        decoded = self.decoder(latent)
        return decoded.unsqueeze(-1)


class CfCAutoencoderAnomalyDetector:
    """
    Reconstruction-based anomaly detector using a CfC (liquid) autoencoder.

    The detector is fitted on normal beats. The anomaly score is the mean
    squared reconstruction error for each beat, which keeps it comparable with
    the PCA, LSTM-autoencoder, and VAE detectors.
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
        self.model = CfCAutoencoderTorch(seq_len, hidden_size, latent_dim).to(self.device)
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
