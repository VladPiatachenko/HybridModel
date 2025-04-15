# HybridModel: Drone Attack Detection with Feature Extractors + ML Classifiers

This project explores **hybrid architectures** that combine deep learning feature extractors with traditional machine learning classifiers to detect drone attacks from time-series sensor data.

## 📌 Objective
To compare various combinations of neural feature extractors (LSTM, CNN1D, Autoencoder) and classifiers (XGBoost, SVM, KNN, Logistic Regression) in the task of binary classification: `attack` vs `no attack`.

## 🗂️ Project Structure
```
HybridModel/
├── dataset/                    ← CSV data with features and labels
├── extractors/                ← Neural feature extractors
│   ├── lstm_extractor.py
│   ├── cnn1d_extractor.py
│   └── autoencoder_extractor.py
├── classifiers/               ← ML classifiers
│   ├── xgboost_classifier.py
│   ├── knn_classifier.py
│   ├── svm_classifier.py
│   └── logreg_classifier.py
├── results/                   ← Evaluation results (CSV + plots)
├── train_and_evaluate.py     ← Main loop: all combinations
└── README.md
```

## 🔍 Data Format
The dataset is expected to be a CSV with the last column as the binary label:
```
feature_1, feature_2, ..., feature_N, label
```
Label:
- `0` = no attack
- `1` = attack

## 🧠 Feature Extractors
| Name       | Architecture                          |
|------------|----------------------------------------|
| `LSTM`     | Bidirectional LSTM + Dropout           |
| `CNN1D`    | Conv1D + BatchNorm + MaxPooling + GAP |
| `Autoenc.` | LSTM encoder + Dense bottleneck        |

## 🤖 Classifiers
| Classifier       | Notes                               |
|------------------|-------------------------------------|
| XGBoost          | With `scale_pos_weight=5`           |
| SVM              | With `class_weight='balanced'`      |
| KNN              | With oversampled attacks            |
| LogisticRegression | With `class_weight='balanced'`    |

## ⚙️ How to Run
Run all experiments:
```bash
python train_and_evaluate.py
```

Results will be saved to `results/scores.csv` and confusion matrices as PNG images.

## 📊 Evaluation Metrics
Each run reports:
- Accuracy
- F1-score
- Precision
- Recall
- `attack_detection_rate` (TPR for class 1)
- `false_attack_rate` (FPR for class 0)

## 📈 Example Output
Results are visualized using boxplots and tables to highlight how:
- Feature extractors affect classifier performance
- Balancing and stride improve generalization

---
## 🔬 Goal
This study demonstrates the **effectiveness of modular hybrid systems** for anomaly detection in drone systems — with a pipeline flexible enough to test more classifiers or real-time deployment models.

