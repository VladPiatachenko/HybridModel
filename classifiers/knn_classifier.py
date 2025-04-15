from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# KNN класифікатор
def classify_and_score(X_train, y_train, X_test, y_test):
    # Ручне oversampling атак (клас 1)
    attack_indices = np.where(y_train == 1)[0]
    non_attack_indices = np.where(y_train == 0)[0]

    n_attack = len(attack_indices)
    n_non_attack = len(non_attack_indices)

    if n_attack < n_non_attack:
        oversample_factor = n_non_attack // n_attack
        extra = n_non_attack % n_attack
        replicated = np.hstack([np.tile(attack_indices, oversample_factor), np.random.choice(attack_indices, extra, replace=False)])

        X_train_bal = np.vstack([X_train[non_attack_indices], X_train[replicated]])
        y_train_bal = np.hstack([y_train[non_attack_indices], y_train[replicated]])
    else:
        X_train_bal, y_train_bal = X_train, y_train

    # Перемішуємо
    shuffled = np.random.permutation(len(y_train_bal))
    X_train_bal = X_train_bal[shuffled]
    y_train_bal = y_train_bal[shuffled]

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_bal, y_train_bal)

    y_pred = model.predict(X_test)

    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)

    # Матриця впевненості
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Attack', 'Attack'], yticklabels=['No Attack', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_knn.png")
    plt.close()

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "attack_detection_rate": recall * 100,
        "false_attack_rate": (cm[0][1] / sum(cm[0])) * 100
    }

    return metrics
