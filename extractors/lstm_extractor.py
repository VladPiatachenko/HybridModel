import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.optimizers import Adam

# LSTM екстрактор ознак
def extract_features(X_train_seq, X_test_seq):
    X_train_seq = np.array(X_train_seq)
    X_test_seq = np.array(X_test_seq)

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    inp = Input(shape=input_shape)
    lstm_out = LSTM(64)(inp)
    model = Model(inputs=inp, outputs=lstm_out)
    model.compile(optimizer=Adam(), loss='mse')

    # Штучне тренування (не обов'язкове, лише для ініціалізації ваг)
    dummy_y = np.zeros((len(X_train_seq), 64))
    model.fit(X_train_seq, dummy_y, epochs=1, batch_size=32, verbose=0)

    # Отримання векторів ознак
    X_train_vec = model.predict(X_train_seq)
    X_test_vec = model.predict(X_test_seq)

    return X_train_vec, X_test_vec
