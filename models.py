import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


class IsolationForestAnomalyDetector:
    """
    Isolation Forest for ECG anomaly detection.
    Fast baseline method that doesn't require labeled data.
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
    
    def fit(self, X_train):
        """Fit Isolation Forest on training data."""
        # X_train shape: (n_samples, features)
        if len(X_train.shape) == 3:
            # If 3D (n_samples, seq_len, n_features), flatten
            X_train = X_train.reshape(X_train.shape[0], -1)
        elif len(X_train.shape) == 2 and X_train.shape[1] == 1:
            # If 2D with single feature
            pass
        
        self.model.fit(X_train)
        return self
    
    def predict(self, X_test):
        """Predict anomalies (-1 = anomaly, 1 = normal)."""
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        return self.model.predict(X_test)
    
    def anomaly_score(self, X_test):
        """Get anomaly scores (higher = more anomalous)."""
        if len(X_test.shape) == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        return -self.model.score_samples(X_test)


class ARMAModel:
    """
    ARMA (AutoRegressive Moving Average) model for anomaly detection.
    Fits an ARIMA model and detects anomalies based on prediction error.
    """
    
    def __init__(self, p=1, d=0, q=1, threshold_percentile=95):
        self.p = p
        self.d = d
        self.q = q
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        
    def fit(self, X_train):
        """Fit ARIMA model on training data."""
        # X_train shape: (n_samples,)
        signal = X_train.flatten()
        self.model = ARIMA(signal, order=(self.p, self.d, self.q))
        self.model = self.model.fit()
        
        # Calculate threshold on training errors
        predictions = self.model.fittedvalues
        errors = np.abs(signal - predictions)
        self.threshold = np.percentile(errors, self.threshold_percentile)
        
        return self
    
    def predict(self, X_test):
        """Predict anomalies (1 = normal, -1 = anomaly)."""
        signal = X_test.flatten()
        predictions = self.model.get_forecast(steps=len(signal)).predicted_mean.values
        
        # Pad if needed
        if len(predictions) < len(signal):
            predictions = np.concatenate([self.model.fittedvalues[-len(signal) + len(predictions):], predictions])
        
        errors = np.abs(signal - predictions)
        anomalies = np.where(errors > self.threshold, -1, 1)
        
        return anomalies


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


class LSTMAutoencoder(keras.Model):
    """
    LSTM-based Autoencoder for temporal ECG anomaly detection.
    """
    
    def __init__(self, input_shape, latent_dim=8):
        super(LSTMAutoencoder, self).__init__()
        self.input_shape_val = input_shape
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(32, activation='relu', return_sequences=True),
            layers.LSTM(latent_dim, activation='relu', return_sequences=False),
        ])
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.RepeatVector(input_shape[0]),
            layers.LSTM(latent_dim, activation='relu', return_sequences=True),
            layers.LSTM(32, activation='relu', return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_shape[1])),
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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


