import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Покращений LSTM екстрактор ознак, який тренується supervised

def extract_features(X_train_seq, y_train, X_test_seq):
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    inp = Input(shape=input_shape)

    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', name="feature_output")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train_seq, y_train, epochs=10, batch_size=32, verbose=0)

    encoder = Model(inputs=inp, outputs=model.get_layer("feature_output").output)
    X_train_vec = encoder.predict(X_train_seq)
    X_test_vec = encoder.predict(X_test_seq)

    return X_train_vec, X_test_vec
