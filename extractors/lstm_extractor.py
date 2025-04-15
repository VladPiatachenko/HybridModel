import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Покращений LSTM екстрактор ознак
def extract_features(X_train_seq, X_test_seq, y_train, y_test):
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    inp = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')

    model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
              epochs=10, batch_size=32, verbose=0)

    X_train_vec = model.predict(X_train_seq)
    X_test_vec = model.predict(X_test_seq)

    return X_train_vec, X_test_vec
