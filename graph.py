import os
import pandas as pd
import matplotlib.pyplot as plt


print(os.getcwd())

# Folder containing all result CSVs
result_dir = "vit_results"




# All model result files (assume CSVs named like vit_basic.csv, vit_overlap.csv, etc.)
model_files = {
    'ViT-Basic': 'vit_basic.csv',
    'ViT-Overlap': 'vit_overlap.csv',
    # Add more entries if needed
}

# Assign a unique color for each model
colors = {
    'ViT-Basic': 'blue',
    'ViT-Overlap': 'green',
    # Add colors for more models
}

# Metrics to plot
metrics = ['Train Loss', 'Test Loss', 'Train Acc', 'Test Acc']

# Plot each metric
for metric in metrics:
    plt.figure(figsize=(10, 6))

    for model_name, filename in model_files.items():
        file_path = os.path.join(result_dir, filename)

        # Read the CSV, skip second row if it's empty
        df = pd.read_csv(file_path, skiprows=[1])

        # Remove the "Total Training Time" row
        df = df[pd.to_numeric(df['Epoch'], errors='coerce').notnull()]

        # Plot
        plt.plot(df['Epoch'], df[metric], label=model_name, color=colors.get(model_name, None))

    # Plot formatting
    plt.title(f'{metric} Comparison Across Models')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    save_path = os.path.join(result_dir, f'plot_{metric.replace(" ", "_").lower()}.png')
    plt.savefig(save_path)
    plt.show()
