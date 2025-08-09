"""
Getting Started with SpoilerShield in Kaggle.

This notebook helps you get familiar with running the project in Kaggle.
It includes basic setup and a simple test of the environment.
"""

# 1. Basic Setup
# First, let's check what's in our dataset
import os
print("Files in dataset:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 2. Install Requirements
# We'll install packages one by one to see what's needed
!pip install pandas numpy matplotlib seaborn nltk wordcloud

# 3. Import Basic Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 4. Test Data Loading
# Replace 'your_dataset.csv' with your actual data file
try:
    # Try to load a small sample of data
    data_path = '/kaggle/input/spoiler-shield-data/your_dataset.csv'  # Update this path
    df = pd.read_csv(data_path)
    print("\nData loaded successfully!")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData shape:", df.shape)
except Exception as e:
    print("\nError loading data:", str(e))
    print("\nPlease update the data_path to point to your dataset")

# 5. Test Project Code
# Try importing our project modules
try:
    import sys
    sys.path.append('/kaggle/input/spoiler-shield-code/implementation/src')
    
    from eda.text_analyzer import TextAnalyzer
    print("\nSuccessfully imported TextAnalyzer!")
    
    # Try creating an analyzer with sample data
    if 'df' in locals() and 'text' in df.columns and 'label' in df.columns:
        analyzer = TextAnalyzer(df['text'], df['label'])
        stats = analyzer.get_text_statistics()
        print("\nText Statistics:", stats)
    else:
        print("\nSample data not available. Please load your dataset first.")
except Exception as e:
    print("\nError importing project code:", str(e))
    print("\nPlease make sure your code dataset is properly uploaded")

# 6. Test Visualization
try:
    if 'analyzer' in locals():
        from eda.visualization import TextVisualizer
        visualizer = TextVisualizer(df['text'], df['label'])
        print("\nSuccessfully created TextVisualizer!")
        
        # Try a simple plot
        visualizer.plot_word_frequencies(n_top=10)
        plt.title("Test Plot - Top 10 Words")
        plt.show()
except Exception as e:
    print("\nError with visualization:", str(e))

# 7. Next Steps
print("\nNext steps:")
print("1. Update the data_path to point to your dataset")
print("2. Try running the full EDA template")
print("3. Experiment with different visualizations")
print("4. Save any outputs to /kaggle/working/") 