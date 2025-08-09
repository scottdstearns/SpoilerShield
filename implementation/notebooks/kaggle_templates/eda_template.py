"""
Template for Kaggle EDA notebooks.

This template shows how to use the project code in a Kaggle notebook.
"""

# Install required packages
!pip install -q -r /kaggle/input/spoiler-shield-code/requirements.txt

# Import project modules
import sys
sys.path.append('/kaggle/input/spoiler-shield-code/implementation/src')

from eda.text_analyzer import TextAnalyzer
from eda.visualization import TextVisualizer
from utils.kaggle_utils import setup_kaggle_environment, get_kaggle_data_path

# Set up the environment
setup_kaggle_environment()

# Load your data
# Note: Update the path to match your Kaggle dataset
data_path = get_kaggle_data_path() / 'your_dataset.csv'
df = pd.read_csv(data_path)

# Initialize analyzers
analyzer = TextAnalyzer(df['text'], df['label'])
visualizer = TextVisualizer(df['text'], df['label'])

# Example EDA
# 1. Basic statistics
stats = analyzer.get_text_statistics()
print("Text Statistics:", stats)

# 2. Class distribution
visualizer.plot_class_distribution()

# 3. Word frequencies
visualizer.plot_word_frequencies(n_top=20)

# 4. Document length distribution
visualizer.plot_document_length_distribution()

# 5. Word clouds by class
for label in df['label'].unique():
    visualizer.plot_wordcloud(class_label=label)
    plt.show()

# Save any outputs
output_path = get_kaggle_output_path() / 'eda_results'
output_path.mkdir(exist_ok=True)
# Add code to save your results here 