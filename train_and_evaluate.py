import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === ІМПОРТИ ЕКСТРАКТОРІВ ===
import extractors.lstm_extractor as lstm_ext
import extractors.cnn1d_extractor as cnn_ext
import extractors.autoencoder_extractor as ae_ext

# === ІМПОРТИ КЛАСИФІКАТОРІВ ===
import classifiers.xgboost_classifier as xgb_clf
import classifiers.knn_classifier as knn_clf
import classifiers.svm_classifier as svm_clf
import classifiers.logreg_classifier as logreg_clf

# === ЗАВАНТАЖЕННЯ ТА РОЗБИТТЯ ===
df = pd.read_csv("dataset/AllFeatureData.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# === МАСШТАБУВАННЯ ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# === ФОРМУВАННЯ СЕКВЕНЦІЙ ===
def make_sequences(X, y, window_size=10, stride=2):
    X_seq, y_seq = [], []
    for i in range(0, len(X) - window_size, stride):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size-1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train = make_sequences(X_train_scaled, y_train_raw, window_size=10, stride=2)
X_test_seq, y_test = make_sequences(X_test_scaled, y_test_raw, window_size=10, stride=2)

# === ВСІ ЕКСТРАКТОРИ ===
extractors = {
    "LSTM": lstm_ext.extract_features,
    "CNN1D": cnn_ext.extract_features,
    "Autoencoder": ae_ext.extract_features
}

# === ВСІ КЛАСИФІКАТОРИ ===
classifiers = {
    "XGBoost": xgb_clf.classify_and_score,
    "KNN": knn_clf.classify_and_score,
    "SVM": svm_clf.classify_and_score,
    "LogReg": logreg_clf.classify_and_score
}

results = []

# === ЦИКЛ ПЕРЕБОРУ ВСІХ КОМБІНАЦІЙ ===
for ext_name, extract_fn in extractors.items():
    print(f"[Extracting features with {ext_name}]")
    X_train_vec, X_test_vec = extract_fn(X_train_seq, X_test_seq)

    for clf_name, clf_fn in classifiers.items():
        print(f"--> Classifying with {clf_name}")
        metrics = clf_fn(X_train_vec, y_train, X_test_vec, y_test)
        results.append({
            "Extractor": ext_name,
            "Classifier": clf_name,
            **metrics
        })

# === ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ ===
results_df = pd.DataFrame(results)
results_df.to_csv("results/scores.csv", index=False)
print("Готово! Результати збережено у results/scores.csv")