# SpoilerShield Model Training Report

**Training Date:** 2025-08-06T07:59:52.451067

## Model Performance Summary

| Model | F1 Score | Accuracy | Precision | Recall (Sensitivity) | Specificity | ROC-AUC |
|-------|----------|----------|-----------|---------------------|-------------|--------|
| Random Forest | 0.4808 | 0.6897 | 0.4294 | 0.5461 | 0.7410 | 0.7124 |
| SVM | 0.4165 | 0.2630 | 0.2630 | 1.0000 | 0.0000 | 0.5739 |
| Baseline_TF-IDF_LogReg | 0.4101 | 0.7709 | 0.6354 | 0.3028 | 0.9380 | 0.7452 |
| Naive Bayes | 0.1891 | 0.7529 | 0.6901 | 0.1096 | 0.9824 | 0.7173 |

**Best Model:** Random Forest
**Best F1 Score:** 0.4808

## Hyperparameter Search Results

**Best Parameters:** {'skipped': True}
**Best Cross-Validation Score:** 0.4101

## Files Created

- /Users/scottstearns/Library/CloudStorage/GoogleDrive-stearns.scottd@gmail.com/My Drive/IK Course Materials/SpoilerShield/outputs/baseline_model.pkl
- skipped
- /Users/scottstearns/Library/CloudStorage/GoogleDrive-stearns.scottd@gmail.com/My Drive/IK Course Materials/SpoilerShield/outputs/random_forest_model.pkl
- /Users/scottstearns/Library/CloudStorage/GoogleDrive-stearns.scottd@gmail.com/My Drive/IK Course Materials/SpoilerShield/outputs/svm_model.pkl
- /Users/scottstearns/Library/CloudStorage/GoogleDrive-stearns.scottd@gmail.com/My Drive/IK Course Materials/SpoilerShield/outputs/naive_bayes_model.pkl
