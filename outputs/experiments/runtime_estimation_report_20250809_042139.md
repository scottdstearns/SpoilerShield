# Runtime Estimation Report

**Generated:** 2025-08-09 04:21:39

## 📊 GridSearch Runtime Estimates

| Sample Size | Single Eval (s) | Total Evals | Est. Time (min) | F1 | AUC |
|-------------|-----------------|-------------|-----------------|-------|-----|
| 10,000 | 2.4 | 144 | 2.8 | 0.685 | 0.747 |
| 20,000 | 4.9 | 144 | 5.7 | 0.664 | 0.735 |
| 40,000 | 10.3 | 144 | 11.4 | 0.673 | 0.746 |
| 60,000 | 12.6 | 144 | 17.0 | 0.627 | 0.748 |

## 🧬 GA Strict Mode Runtime Estimates

| Sample Size | Single Eval (s) | Total Evals | Est. Time (min) |
|-------------|-----------------|-------------|----------------|
| 10,000 | 2.5 | 600 | 29.7 |
| 20,000 | 5.0 | 600 | 59.4 |
| 40,000 | 10.5 | 600 | 118.7 |
| 60,000 | 12.6 | 600 | 178.1 |

## 🧬 GA Advantage Mode Runtime Estimates

| Sample Size | Single Eval (s) | Total Evals | Est. Time (min) |
|-------------|-----------------|-------------|----------------|
| 10,000 | 2.4 | 600 | 13.4 |
| 20,000 | 5.0 | 600 | 26.9 |
| 40,000 | 10.2 | 600 | 53.8 |
| 60,000 | 12.7 | 600 | 80.7 |

## 🎯 Recommendations

**Recommended Sample Size:** 20,000

**Estimated Runtimes:**
- GridSearch: 5.7 minutes
- GA Strict: 59.36666666666667 minutes
- GA Advantage: 26.9 minutes
- **Total: 91.9 minutes**

**Parameter Configuration:**
- Grid combinations: 48
- CV folds: 3
- GA population: 20
- GA Strict generations: 28
- GA Advantage generations: 63

**GA Advantage Analysis:**
- Estimated generations: 63
- Likelihood of advantage: HIGH
- Reasoning: With 63 generations, GA has sufficient opportunity to explore continuous parameter space and demonstrate optimization advantage over discrete grid.

**Expected Performance:**
- F1 Score: 0.664
- ROC AUC: 0.735

