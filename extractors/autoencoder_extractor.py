import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam

# Autoencoder екстрактор ознак (LSTM-based)
def extract_features(X_train_seq, X_test_seq):
    X_train_seq = np.array(X_train_seq)
    X_test_seq = np.array(X_test_seq)

    timesteps = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    # Вхід
    inp = Input(shape=(timesteps, n_features))

    # Енкодер
    encoded = LSTM(64)(inp)

    # Декодер (для навчання)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(n_features, return_sequences=True)(decoded)

    # Autoencoder модель
    autoencoder = Model(inputs=inp, outputs=decoded)
    autoencoder.compile(optimizer=Adam(), loss='mse')

    # Навчаємо autoencoder (unsupervised)
    autoencoder.fit(X_train_seq, X_train_seq, epochs=10, batch_size=32, verbose=0)

    # Екстрактор ознак (тільки encoder)
    encoder = Model(inputs=inp, outputs=encoded)

    X_train_vec = encoder.predict(X_train_seq)
    X_test_vec = encoder.predict(X_test_seq)

    return X_train_vec, X_test_vec
