# PoC A/B: GridSearch vs GA (LogReg only)

**Budget parity:** 45 CV evaluations (3-fold × 15 combos)  
**Same folds:** Yes (fixed StratifiedKFold indices)

## Results (held-out test F1)

- **GridSearch (strict grid)**: F1=0.5296, ROC AUC=0.7488
- **GA (strict A/B, discrete grid)**: F1=0.5278, Time=270.0335s

## Best Params

- **GridSearch**: `{'classifier__C': 0.5, 'classifier__penalty': 'l2', 'tfidf__max_features': 10000}`
- **GA (strict)**: `{'tfidf__max_features': 40000, 'classifier__C': 1, 'classifier__penalty': 'l2'}`

## Performance Comparison

| Method | Test F1 | CV F1 | Time (s) | Evaluations |
|--------|---------|-------|----------|-------------|
| GridSearch | 0.5296 | 0.5305 | 32.1941 | 45 |
| GA (strict) | 0.5278 | 0.5272 | 270.0335 | 45 |

## Key Findings

**Winner by F1 Score**: GridSearch
**Efficiency Winner**: GridSearch

## Notes
- GA uses tournament selection + uniform crossover + adaptive mutation
- Budget parity ensures fair comparison (same CV folds, same evaluation count)
- This is a proof-of-concept for larger-scale A/B testing

