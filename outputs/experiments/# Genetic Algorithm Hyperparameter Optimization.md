# Genetic Algorithm Hyperparameter Optimization – Project Summary

## Context
This project implements and evaluates a **Genetic Algorithm (GA)** for hyperparameter optimization, comparing it fairly to a traditional **GridSearchCV** baseline.

The primary goal is to determine whether GA can match or exceed GridSearch performance under **identical compute budgets** and **identical cross-validation folds**, making the results directly comparable and suitable for rigorous, compliance-friendly contexts such as regulated industries (e.g., medical devices).

---

## Objectives
1. **Fair apples-to-apples benchmarking** between GA and GridSearch.
2. **Strict budget parity**: GA is allowed the same number of cross-validation evaluations as GridSearch.
3. **Identical validation folds**: GA and GridSearch share the same StratifiedKFold splits to remove sampling noise.
4. **Multi-metric evaluation**: Track both F1-score and ROC AUC on the same runs.
5. **Transparent evolution tracking**: Best-so-far curves and per-generation statistics for GA.
6. **Compliance-friendly reporting**: Produce machine-readable JSON, CSV logs, and human-readable Markdown reports.

---

## Engineering Innovations in This Project
While the concept of using GAs for hyperparameter optimization is well-established, this project introduces several thoughtful engineering decisions that are uncommon in typical GA-HPO examples:

- **Budget-Matched Comparisons**  
  GA and GridSearch are allocated exactly the same number of CV evaluations. This is rare in literature, where GA often gets more opportunities.

- **Identical CV Fold Indices**  
  Both methods use the *same* train/validation splits to eliminate variance from data partitioning.

- **Two Evaluation Modes**  
  - **Strict A/B Mode**: GA restricted to the same discrete parameter space as GridSearch.  
  - **Advantage Mode**: GA allowed to search continuous ranges (e.g., C ∈ [1, 5]) while keeping the same eval budget.

- **Multi-Metric Tracking**  
  F1 and ROC AUC are both evaluated per CV trial, with best-so-far curves plotted.

- **Per-Generation Analytics**  
  GA logs best/mean/std of F1 and AUC per generation, along with diversity metrics (`unique_configs`).

- **Compliance-Ready Outputs**  
  JSON for machine parsing, Markdown for human review, PNG plots, and CSV stats for audits.

---

## Prior Art – Seminal GA-HPO References
The concept of applying evolutionary algorithms, including GAs, to hyperparameter optimization has been explored in the literature for decades:

1. **Angeline, P. J., & Pollack, J. B. (1993).** *Evolutionary algorithm optimization of neural network architectures.*  
   Early work demonstrating that evolutionary methods can discover effective NN architectures and hyperparameters.

2. **Young, T., Rose, D., Karnowski, T., Lim, S.-H., & Patton, R. (2015).** *Optimizing deep learning hyper-parameters through an evolutionary algorithm.* Proceedings of the Workshop on Machine Learning in High-Performance Computing Environments.  
   Applies GA to deep learning hyperparameters, showing competitive performance vs. grid/random search.

---

## Related Tools and Frameworks
- **[sklearn-genetic-opt](https://github.com/rodrigo-arenas/Sklearn-genetic-opt)** – A GA-based hyperparameter tuner compatible with scikit-learn estimators.
- **[DEAP](https://deap.readthedocs.io/)** – General-purpose evolutionary computation framework widely used for GA research.

---

## Summary Statement
> This project is not the first to use a Genetic Algorithm for hyperparameter optimization, but it advances the practice by enforcing strict, reproducible, and budget-matched comparisons with GridSearch, using identical validation folds and multi-metric tracking. The design is tailored for compliance-heavy environments where transparency, reproducibility, and fairness of evaluation are critical. The resulting framework is both a research tool for method comparison and a practical system for fair model selection in regulated machine learning workflows.