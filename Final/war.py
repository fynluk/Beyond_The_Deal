import logging
import os
import numpy as np
import pandas as pd
import lseg.data as ld
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

UNIVERSE = "0#.STOXX"
FREQ     = "W"   # "W" oder "M"

# >>> HIER DEINE 3 ZEITRÄUME DEFINIEREN <<<
PERIODS = {
    "Period_1": ("2022-01-01", "2022-06-30"),
    "Period_2": ("2022-01-01", "2022-12-31"),
    "Period_3": ("2022-01-01", "2023-12-31"),
    "Period_4": ("2021-01-01", "2025-12-31")
}

# ---------------------------------------------------
# REFINITIV SESSION
# ---------------------------------------------------

def refinitiv_session():
    ld.open_session()

# ---------------------------------------------------
# DATA
# ---------------------------------------------------

def get_universe(end_date):
    query = ld.get_data(
        universe=UNIVERSE,
        fields=["TR.RIC"],
        parameters={"SDate": end_date}
    )
    return set(query["Instrument"].dropna())

def get_prices(universe, start_date, end_date):

    prices = ld.get_data(
        universe=universe,
        fields=["TR.PriceClose","TR.PriceClose.date"],
        parameters={
            "Frq": FREQ,
            "SDate": start_date,
            "EDate": end_date
        }
    )

    prices.columns = ["Instrument","Price","Date"]
    prices = prices.dropna()

    prices = prices.groupby(["Date","Instrument"]).mean().reset_index()
    prices = prices.pivot(index="Date", columns="Instrument", values="Price")

    return prices

# ---------------------------------------------------
# RETURN & RISK
# ---------------------------------------------------

def compute_metrics(prices):

    scale = 52 if FREQ == "W" else 12

    returns = prices.pct_change(fill_method=None)

    mean_return = returns.mean() * scale
    volatility  = returns.std()  * np.sqrt(scale)

    df = pd.DataFrame({
        "Return": mean_return,
        "Risk": volatility
    }).dropna()

    return df

# ---------------------------------------------------
# PLOT
# ---------------------------------------------------

def plot_risk_return(df, label):

    grid_color  = (236/255,237/255,239/255)
    point_color = (0/255,39/255,80/255)
    line_color  = (245/255,158/255,0/255)

    x = df["Risk"].values
    y = df["Return"].values

    plt.figure(figsize=(12,8))
    plt.gca().set_axisbelow(True)

    # Scatter
    plt.scatter(x, y, alpha=0.5, color=point_color)

    # Regression
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res/ss_tot

    x_line = np.linspace(0, 1, 200)
    y_line = slope * x_line + intercept

    plt.plot(x_line, y_line, color=line_color, linewidth=2)

    plt.text(
        0.05, 0.95,
        f"Slope = {slope:.4f}\nR² = {r_squared:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=12
    )

    plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    plt.xlabel("Risk (Volatility)", fontsize=16)
    plt.ylabel("Annualized Return", fontsize=16)

    plt.title(label, fontsize=16)

    plt.grid(True, color=grid_color)

    os.makedirs("Plots", exist_ok=True)
    plt.savefig(f"Plots/RiskReturn_{label}.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    refinitiv_session()

    for label, (start_date, end_date) in PERIODS.items():

        print(f"Running {label}: {start_date} → {end_date}")

        universe = get_universe(end_date)
        prices = get_prices(universe, start_date, end_date)

        metrics = compute_metrics(prices)

        plot_risk_return(metrics, label)

    ld.close_session()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
