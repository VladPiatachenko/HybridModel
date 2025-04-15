import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# Покращений Autoencoder екстрактор ознак (LSTM-based)
def extract_features(X_train_seq, X_test_seq):
    X_train_seq = np.array(X_train_seq)
    X_test_seq = np.array(X_test_seq)

    timesteps = X_train_seq.shape[1]
    n_features = X_train_seq.shape[2]

    # Вхід
    inp = Input(shape=(timesteps, n_features))

    # Енкодер
    x = LSTM(128, return_sequences=False)(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    bottleneck = Dense(64, activation='relu')(x)

    # Декодер
    x = RepeatVector(timesteps)(bottleneck)
    x = LSTM(n_features, return_sequences=True)(x)

    autoencoder = Model(inputs=inp, outputs=x)
    autoencoder.compile(optimizer=Adam(), loss='mse')

    autoencoder.fit(X_train_seq, X_train_seq, epochs=20, batch_size=32, verbose=0)

    encoder = Model(inputs=inp, outputs=bottleneck)

    X_train_vec = encoder.predict(X_train_seq)
    X_test_vec = encoder.predict(X_test_seq)

    return X_train_vec, X_test_vec