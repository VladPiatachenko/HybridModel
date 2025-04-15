import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Покращений LSTM екстрактор ознак
def extract_features(X_train_seq, X_test_seq):
    X_train_seq = np.array(X_train_seq)
    X_test_seq = np.array(X_test_seq)

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    inp = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=False))(inp)
    x = Dropout(0.2)(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(), loss='mse')

    dummy_y = np.zeros((len(X_train_seq), model.output_shape[1]))
    model.fit(X_train_seq, dummy_y, epochs=10, batch_size=32, verbose=0)

    X_train_vec = model.predict(X_train_seq)
    X_test_vec = model.predict(X_test_seq)

    return X_train_vec, X_test_vec
