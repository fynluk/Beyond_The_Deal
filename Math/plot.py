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


def show_interval(interval_dict, stock):
    # Sort Keys & Values
    sorted_items = sorted(interval_dict.items())
    x_vals = [k for k, _ in sorted_items]
    y_vals = [v for _, v in sorted_items]

    # Create Plot
    plt.figure(figsize=(8, 5))
    plt.axvline(x=0, color='#002750', linestyle='--', label='Zieldatum (0)')
    plt.plot(x_vals, y_vals, marker='.', linestyle='-', color='#FF9E00', label="Close")

    plt.title(f"Share Price: {stock.name}")
    plt.xlabel('Days relative to the Announcement Date')
    plt.ylabel("Share Price (Open!)")

    interval_range = max(x_vals) - min(x_vals)
    step = max(1, interval_range // 10)  # maximal 10 Ticks auf der x-Achse
    plt.xticks(ticks=[x for x in range(min(x_vals), max(x_vals) + 1) if x % step == 0])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()