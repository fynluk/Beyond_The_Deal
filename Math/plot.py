import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from collections import Counter
from collections import defaultdict

from matplotlib import ticker


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

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(years, number_of_deals, color='#FF9E00', alpha=0.8, label='Number of Deals')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Deals')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 75000)
    ax1.set_yticks(range(0, 75001, 10000))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax2 = ax1.twinx()
    ax2.plot(years, deal_volume, color='#002750', label='Transaction Volume in bil. USD')
    ax2.set_ylabel('Transaction Volume in bil. USD')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 6000)
    ax2.set_yticks(range(0, 6001, 1000))
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    #plt.title('asdf')
    plt.tight_layout()
    plt.grid(True)
    plt_patch1 = mpatches.Patch(color='#FF9E00', label='Number of Deals')
    plt_patch2 = mpatches.Patch(color='#002750', label='Transaction Volume')
    plt.legend(handles=[plt_patch1, plt_patch2], loc='upper left')
    plt.show()


def show_returns(returns, deals):
    if len(returns) != len(deals):
        raise ValueError("Beide Listen müssen die gleiche Länge haben.")

        # Gruppiere Werte nach Jahr
    yearly_data = defaultdict(list)
    for value, deal in zip(returns, deals):
        year = deal.announcement_date.year
        yearly_data[year].append(float(value))  # Decimal → float

    # Sortiere die Jahre
    sorted_years = sorted(yearly_data.keys())
    data_per_year = [yearly_data[year] for year in sorted_years]

    # Zeichne Boxplot
    plt.figure(figsize=(12, 6))
    bplot = plt.boxplot(data_per_year, labels=sorted_years, patch_artist=True)

    # Formatierung
    plt.xlabel("Jahr")
    plt.ylabel("Wert in %")
    plt.title("Verteilung der Werte pro Jahr (Boxplot)")
    plt.grid(True, axis='y')

    # Y-Achse auf Prozentformat setzen
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # 1.0 == 100%

    plt.tight_layout()
    plt.show()