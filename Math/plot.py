import matplotlib.pyplot as plt
from collections import Counter

#TODO ADD Diagramm  x-Axis: P&L on Share Price after aDate in % for the Target
#                   y-Axis: P&L on Share Price after aDate in % for the Buyer -> Any Findings?

def plot_columns(data, cons):
    # Count name frequencies
    name_counts = Counter(data)
    print(name_counts)

    # Extract names and their frequencies
    name_list = list(name_counts.keys())
    frequencies = list(name_counts.values())

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.bar(name_list, frequencies, color='skyblue')
    plt.axhline(y=cons, color='red', linestyle='--', label=f'Constant = {cons}')
    plt.xlabel('Names')
    plt.ylabel('Frequency')
    plt.title('Name Frequency')
    plt.xticks(rotation=90)  # Rotate labels vertically
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()