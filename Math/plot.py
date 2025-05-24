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


def imaa(data):
    years = [entry["year"] for entry in data]
    number_of_deals = [entry["number_of_deals"] for entry in data]
    deal_volume = [entry["ma_value_billion_usd"] for entry in data]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.bar(years, number_of_deals, color='#FF9E00', alpha=0.8)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Deals')
    ax1.tick_params(axis='y', labelcolor='#002750')
    ax1.set_ylim(0, 75000)
    ax1.set_yticks(range(0, 75001, 10000))

    ax2 = ax1.twinx()
    ax2.plot(years, deal_volume, color='#002750')
    ax2.set_ylabel('Transaction Volume in bil. USD')
    ax2.tick_params(axis='y', labelcolor='#002750')
    ax2.set_ylim(0, 6000)
    ax2.set_yticks(range(0, 6001, 1000))

    #plt.title('asdf')
    plt.tight_layout()
    plt.grid(True)
    plt.show()