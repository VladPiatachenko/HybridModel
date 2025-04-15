from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# SVM класифікатор
def classify_and_score(X_train, y_train, X_test, y_test):
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

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
    plt.savefig("results/confusion_matrix_svm.png")
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
