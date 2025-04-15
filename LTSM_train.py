import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Дані ===
df = pd.read_csv("dataset/AllFeatureData.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# === Розбиття ===
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# === Секвенції ===
def make_sequences(X, y, window_size=10, stride=2):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - window_size, stride):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size-1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train = make_sequences(X_train_scaled, y_train_raw)
X_test_seq, y_test = make_sequences(X_test_scaled, y_test_raw)

# === LSTM екстрактор + тренування ===
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
inp = Input(shape=input_shape)
x = Bidirectional(LSTM(64, return_sequences=True))(inp)
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

out = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inp, outputs=out)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
          epochs=10, batch_size=32, verbose=1)

# === Оцінка ===
y_pred_proba = model.predict(X_test_seq)
y_pred = (y_pred_proba > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\nF1-score: {f1:.3f}")

# === Матриця впевненості ===
cm = confusion_matrix(y_test, y_pred)
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Attack', 'Attack'],
            yticklabels=['No Attack', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("results/confusion_matrix_lstm_classifier.png")
plt.close()

# === Збереження результатів ===
results = pd.DataFrame([{"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}])
results.to_csv("results/scores_lstm_only.csv", index=False)
print("\nРезультати збережено у results/")
