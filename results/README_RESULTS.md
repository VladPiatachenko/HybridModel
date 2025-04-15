# 📁 Results Summary

This folder contains evaluation results for the hybrid anomaly detection experiments.

---

## 📄 `scores.csv`
This CSV contains the main benchmarking table for all extractor+classifier combinations.  
Each row includes:

- `Extractor`: feature extractor used (LSTM, CNN1D, Autoencoder)
- `Classifier`: ML model used (XGBoost, SVM, KNN, LogReg)
- `accuracy`: standard accuracy
- `precision`, `recall`, `f1`: classification metrics
- `attack_detection_rate`: True Positive Rate for attacks (class 1)
- `false_attack_rate`: False Positive Rate (false alarms)

---

## 📊 `confusion_matrix_*.png`
Each file is the confusion matrix of the corresponding model, useful to visualize:
- False positives (false alarms)
- False negatives (missed attacks)

---

## 📈 Notes on Versioning
Experiments were run in three stages:
1. **Baseline** — default extractors, no balancing
2. **Stride** — added overlapping windows (`stride=2`)
3. **Balanced** — added `class_weight`, oversampling, tuned architectures

You can match results by comparing CSVs:
- `scores.csv` — final
- Older ones stored manually (`scores (1).csv`, etc.)

---

## 🧠 Recommendations
- `LSTM + SVM` gives best trade-off between recall and false alarms
- `Autoencoder + LogReg` achieves very high detection, but with high FP
- `CNN1D` needs further tuning to compete with LSTM-based models
- Further improvements can include ensemble classifiers or attention-based encoders

---

✅ Use these results to justify your design choices or to inspire next-stage experimentation!
