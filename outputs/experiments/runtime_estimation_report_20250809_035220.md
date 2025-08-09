# Runtime Estimation Report

**Generated:** 2025-08-09 03:52:20

## 📊 GridSearch Runtime Estimates

| Sample Size | Single Eval (s) | Total Evals | Est. Time (min) | F1 | AUC |
|-------------|-----------------|-------------|-----------------|-------|-----|
| 1,000 | 0.2 | 144 | 0.5 | 0.627 | 0.693 |
| 2,000 | 0.5 | 144 | 1.2 | 0.629 | 0.688 |
| 5,000 | 1.2 | 144 | 2.8 | 0.661 | 0.739 |
| 10,000 | 2.5 | 144 | 5.9 | 0.685 | 0.747 |

## 🧬 Genetic Algorithm Runtime Estimates

| Sample Size | Single Eval (s) | Total Evals | Parallel Factor | Est. Time (min) |
|-------------|-----------------|-------------|-----------------|----------------|
| 1,000 | 0.2 | 160 | 0.40 | 1.3 |
| 2,000 | 0.5 | 160 | 0.40 | 3.3 |
| 5,000 | 1.2 | 160 | 0.40 | 7.7 |
| 10,000 | 2.5 | 160 | 0.40 | 16.8 |

## 🎯 Recommendations

**Recommended Sample Size:** 10,000

**Estimated Runtimes:**
- GridSearch: 5.9 minutes
- Genetic Algorithm: 16.8 minutes
- **Total: 22.8 minutes**

**Parameter Configuration:**
- Grid combinations: 12
- CV folds: 3
- GA population: 10
- GA generations: 6

**Expected Performance:**
- F1 Score: 0.685
- ROC AUC: 0.747

