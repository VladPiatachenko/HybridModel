import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

# 1D CNN екстрактор ознак
def extract_features(X_train_seq, X_test_seq):
    X_train_seq = np.array(X_train_seq)
    X_test_seq = np.array(X_test_seq)

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    inp = Input(shape=input_shape)

    # CNN блок: витягуємо локальні патерни в часі
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inp)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = GlobalAveragePooling1D()(x)  # стискаємо до вектору ознак

    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(), loss='mse')

    # Штучне тренування (просто для ініціалізації)
    dummy_y = np.zeros((len(X_train_seq), 64))
    model.fit(X_train_seq, dummy_y, epochs=1, batch_size=32, verbose=0)

    X_train_vec = model.predict(X_train_seq)
    X_test_vec = model.predict(X_test_seq)

    return X_train_vec, X_test_vec
