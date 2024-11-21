import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find all CSV files in the current directory and subdirectories
csv_files = glob.glob('**/*.csv', recursive=True)

# Plot 'perplexity POS' vs 'training iterations' for all CSV files
plt.figure(figsize=(10, 6))

for file in csv_files:
    data = pd.read_csv(file)
    plt.plot(data['training iterations'], data['perplexity POS'], label=os.path.basename(file))

plt.xlabel('Training Iterations')
plt.ylabel('Perplexity POS')
plt.title('Perplexity POS vs. Training Iterations')
plt.legend()
plt.show()

# Plot 'perplexity' vs 'training iterations' for all CSV files
plt.figure(figsize=(10, 6))

for file in csv_files:
    data = pd.read_csv(file)
    plt.plot(data['training iterations'], data['perplexity'], label=os.path.basename(file))

plt.xlabel('Training Iterations')
plt.ylabel('Perplexity')
plt.title('Perplexity vs. Training Iterations')
plt.legend()
plt.show()
