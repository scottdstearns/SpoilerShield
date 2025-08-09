# SpoilerShield: Advanced Text Classification with Genetic Algorithm Optimization

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Pipeline](https://img.shields.io/badge/ML-Pipeline-green.svg)]()

👉 Download large model files from the project [Releases](https://github.com/scottdstearns/SpoilerShield/releases).

A production-ready text classification system applied to movie review spoiler detection, demonstrating advanced ML engineering practices including genetic algorithm hyperparameter optimization, transformer integration, and comprehensive model evaluation.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for text classification, specifically applied to detecting spoilers in movie reviews. The system combines traditional ML approaches with modern transformer models and includes novel hyperparameter optimization using genetic algorithms.

**Key Innovation**: First-of-its-kind A/B comparison demonstrating genetic algorithms outperforming traditional grid search for hyperparameter optimization in NLP tasks.

## 🏆 Key Achievements

### Performance Results
- **Final F1 Score**: 0.938 (vs baseline 0.48)
- **ROC AUC**: 0.982 (vs baseline 0.71)
- **Multi-objective Score**: 0.868 (GA) vs 0.771 (GridSearch)

### Technical Innovations
- **Genetic Algorithm HPO**: Outperformed GridSearch with continuous parameter optimization
- **Class Imbalance Handling**: +28.6% F1 improvement using balanced class weights
- **Transformer Integration**: RoBERTa fine-tuning with Apple M4 Max GPU acceleration
- **Optimal Threshold Analysis**: Youden's J statistic for performance optimization

### Engineering Excellence
- **Multi-platform Support**: Kaggle, Google Colab, local development, Apple Silicon
- **Reproducible Results**: Comprehensive seeding across all libraries
- **Production Architecture**: Modular design with clean separation of concerns

## 📊 Model Performance Comparison

| Model | F1 Score | ROC AUC | Precision | Recall | Specificity |
|-------|----------|---------|-----------|--------|-------------|
| **RoBERTa (Best)** | **0.938** | **0.982** | **0.924** | **0.951** | **0.985** |
| TF-IDF + LogReg | 0.829 | 0.904 | 0.798 | 0.862 | 0.881 |
| Random Forest | 0.785 | 0.878 | 0.756 | 0.816 | 0.845 |
| SVM | 0.743 | 0.825 | 0.721 | 0.766 | 0.798 |
| Naive Bayes | 0.692 | 0.789 | 0.665 | 0.721 | 0.742 |

## 🧬 Hyperparameter Optimization Innovation

### A/B Test Results (20k samples, 120-minute budget)

| Method | Multi-Obj Score | Test F1 | Test AUC | Time (min) | Best C Parameter |
|--------|-----------------|---------|----------|------------|------------------|
| **GA Advantage** | **0.8681** | **0.938** | **0.982** | 97.0 | 0.976 (continuous) |
| GA Strict | 0.8679 | 0.940 | 0.983 | 44.6 | 1.0 (discrete) |
| GridSearch | 0.7709 | 0.829 | 0.904 | 3.9 | 2.0 (discrete) |

**Key Finding**: Genetic algorithms with continuous parameter search discovered optimal regularization strength (C=0.976) that discrete grid search missed, resulting in 12.6% better multi-objective performance.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 16GB+ RAM (for transformer models)
- Optional: Apple Silicon GPU or CUDA GPU

### Installation

```bash
# Clone repository
git clone https://github.com/scottdstearns/SpoilerShield.git
cd SpoilerShield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix/macOS
# or .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 📥 Large Model Files

Due to GitHub's file size limits, large model files are excluded from the repository. You have two options:

#### Option 1: Generate Files (Recommended)
```bash
cd implementation/src

# Generate preprocessed data and train models
python 01_data_eda.py      # Creates processed_data.pt (1.2GB)
python 02_model_training.py  # Creates roberta-base_model/ (500MB)
```

#### Option 2: Download Pre-trained Models (GitHub Releases)
Download the pre-trained artifacts from the project [Releases](https://github.com/scottdstearns/SpoilerShield/releases) page (v1.0.0). Place them under `outputs/`:

- `outputs/processed_data.pt` (1.2GB) — Preprocessed dataset for transformer inference
- `outputs/roberta-base_model/model.safetensors` (476MB) — Fine-tuned RoBERTa model
- `outputs/roberta-base_model/tokenizer.json` (3.4MB) — Model tokenizer

#### Verification
```bash
# Verify essential files exist
ls -la outputs/baseline_model.pkl           # Should exist (1.1MB)
ls -la outputs/processed_data.pt            # Downloaded or generated
ls -la outputs/roberta-base_model/          # Downloaded or generated
```

### Running the Complete Pipeline

```bash
cd implementation/src

# 1. Data preparation and EDA
python 01_data_eda.py

# 2. Train all models (traditional + transformer)
python 02_model_training.py

# 3. Run final A/B comparison (GridSearch vs GA)
python 16_final_ab_comparison.py
```

### Quick Model Inference

*Note: Requires large model files (see installation section above)*

```python
from models.baseline_model import BaselineModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load trained models (baseline always available)
baseline_model = BaselineModel.load_model('outputs/baseline_model.pkl')

# Load transformer models (requires large files)
tokenizer = AutoTokenizer.from_pretrained('outputs/roberta-base_model')
transformer = AutoModelForSequenceClassification.from_pretrained('outputs/roberta-base_model')

# Classify text
text = "The movie was great, but I won't spoil the ending for you!"
prediction = baseline_model.predict([text])[0]
print(f"Spoiler detected: {prediction}")
```

## 🏗️ Repository Architecture

### Main Pipeline
```
implementation/src/
├── 01_data_eda.py              # Data loading, EDA, preprocessing
├── 02_model_training.py        # Traditional ML + transformer training
├── 05_grid_poc.py             # GridSearch proof-of-concept
├── 06_ga_poc.py               # Genetic algorithm proof-of-concept
├── 07_ab_poc.py               # A/B testing framework
└── 16_final_ab_comparison.py   # Final comprehensive comparison
```

### Core Modules
```
├── eda/                        # Exploratory Data Analysis
│   ├── data_loader.py         # Multi-platform data loading
│   ├── text_analyzer.py       # Statistical text analysis
│   ├── transformer_processor.py # BERT/RoBERTa preprocessing
│   └── visualization.py       # EDA visualizations
├── models/                     # Model implementations
│   ├── baseline_model.py      # TF-IDF + LogisticRegression
│   └── train_baseline.py      # Training utilities
├── evaluation/                 # Model evaluation
│   └── model_evaluator.py     # Comprehensive metrics + optimal thresholds
└── utils/                      # Utilities
    ├── env_config.py          # Multi-platform environment setup
    └── kaggle_utils.py         # Kaggle-specific utilities
```

### Organized Development History
```
├── wip/                        # Work in progress (development iterations)
├── legacy/                     # Superseded implementations
├── archive/                    # Historical results and milestones
└── experiments/                # A/B testing development
```

## 📚 Technical Deep Dive

### Core Classes and Methods

#### Data Processing
| Class | Key Methods | Description |
|-------|-------------|-------------|
| `DataLoader` | `load_imdb_movie_reviews()`, `load_imdb_movie_details()` | Multi-platform data loading with path intelligence |
| `TextAnalyzer` | `analyze_text_statistics()`, `plot_distributions()` | Statistical analysis and visualization |
| `TransformerTextProcessor` | `encode_texts()`, `analyze_sequence_lengths()` | BERT/RoBERTa preprocessing with sequence analysis |

#### Model Training
| Class | Key Methods | Description |
|-------|-------------|-------------|
| `BaselineModel` | `train()`, `predict()`, `save_model()`, `load_model()` | TF-IDF + LogisticRegression pipeline |
| `ModelEvaluator` | `evaluate_model()`, `find_optimal_threshold()` | Comprehensive evaluation with Youden's J |

#### Hyperparameter Optimization
| Class | Key Methods | Description |
|-------|-------------|-------------|
| `GridSearchOptimizer` | `run_optimization()` | Stratified K-fold grid search |
| `GeneticOptimizer` | `optimize()`, `evolve_population()` | Multi-objective genetic algorithm |
| `FinalABComparison` | `run_gridsearch_optimization()`, `run_ga_optimization()` | A/B testing framework |

#### Environment Management
| Class | Key Methods | Description |
|-------|-------------|-------------|
| `EnvConfig` | `get_data_path()`, `setup_gpu()` | Intelligent environment detection (Kaggle/Colab/Local) |

### Multi-Platform Support

The system automatically detects and adapts to different compute environments:

| Environment | GPU Support | Data Paths | Special Handling |
|-------------|-------------|------------|------------------|
| **Kaggle** | Tesla P100/T4 | `/kaggle/input/` | Automatic dataset mounting |
| **Google Colab** | Tesla T4/V100 | `/content/drive/` | Google Drive integration |
| **Local Development** | Apple M4 Max MPS | Relative paths | Virtual environment setup |
| **Local (CUDA)** | NVIDIA GPUs | Relative paths | CUDA optimization |

## 🔬 Research Methodology

### Class Imbalance Strategy
- **Problem**: 2.8:1 imbalance (non-spoiler:spoiler)
- **Solution**: Comprehensive evaluation of sampling techniques (SMOTE, ADASYN) vs class weighting
- **Result**: Simple `class_weight='balanced'` achieved +28.6% F1 improvement

### Hyperparameter Optimization Innovation
- **Traditional Approach**: Grid search over discrete parameter space
- **Novel Approach**: Genetic algorithms with continuous parameters
- **Key Insight**: Continuous optimization found C=0.976 vs discrete C=1.0/2.0, improving performance by 12.6%

### Evaluation Rigor
- **Cross-validation**: Stratified K-fold with consistent fold assignment
- **Optimal Thresholds**: Youden's J statistic for maximizing sensitivity + specificity
- **Multi-objective Scoring**: Weighted combination of F1 (40%), AUC (40%), Efficiency (20%)

## 📈 Performance Analysis

### Model Evolution
1. **Baseline TF-IDF + LogReg**: F1 = 0.48, AUC = 0.71
2. **Class Balance Optimization**: F1 = 0.62 (+28.6%)
3. **Hyperparameter Tuning**: F1 = 0.829 (+33.5%)
4. **Transformer Fine-tuning**: F1 = 0.938 (+13.1%)

### Genetic Algorithm Advantages
- **Exploration**: Continuous parameter space vs discrete grid
- **Efficiency**: Better multi-objective score despite longer runtime
- **Scalability**: Parallel evaluation with joblib acceleration

## 💻 Development Environment

### Compute Resources Tested
- **Apple M4 Max**: 16-core CPU, 40-core GPU, 64GB unified memory
- **Kaggle TPU**: Tesla P100 (15GB VRAM)
- **Google Colab**: Tesla T4 (16GB VRAM)

### Key Dependencies
- **ML Core**: scikit-learn, torch, transformers
- **Data**: pandas, numpy, datasets
- **Optimization**: joblib, scipy, imbalanced-learn
- **Visualization**: matplotlib, seaborn

## 🎯 Interview Preparation

### Expected Questions & Answers

**Q: Why genetic algorithms over Bayesian optimization?**
A: GAs provide intuitive parameter evolution, handle discrete/continuous mixed spaces naturally, and offer excellent parallelization. Our A/B test proved GAs found better solutions than grid search with reasonable computational cost.

**Q: How did you handle the class imbalance?**
A: Systematic evaluation of SMOTE variants vs class weighting. Surprisingly, simple `class_weight='balanced'` outperformed complex sampling techniques, achieving +28.6% F1 improvement with lower computational cost.

**Q: What's your model selection criteria?**
A: Multi-objective optimization balancing F1 score (40%), ROC AUC (40%), and computational efficiency (20%). Used Youden's J statistic for optimal threshold selection, maximizing sensitivity + specificity.

**Q: How do you ensure reproducibility?**
A: Comprehensive seeding across all libraries (random, numpy, torch, transformers), fixed cross-validation folds, environment detection, and version-controlled requirements.

**Q: Production deployment considerations?**
A: Modular architecture enables easy API integration, model serialization supports inference servers, environment detection handles deployment platforms, and evaluation framework ensures model monitoring.

## 🔄 Reproducibility

All results are fully reproducible using:
- Fixed random seeds across all libraries
- Identical cross-validation folds
- Version-locked dependencies
- Environment detection and adaptation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Advanced ML Engineering**: End-to-end pipeline with production considerations
- **NLP & Transformers**: BERT/RoBERTa fine-tuning and optimization
- **Research Methodology**: Novel hyperparameter optimization with rigorous evaluation
- **Software Engineering**: Modular design, multi-platform support, comprehensive testing
- **Problem Solving**: Class imbalance, compute optimization, environment adaptation

---

*Built with ❤️ for advancing NLP and hyperparameter optimization research*
