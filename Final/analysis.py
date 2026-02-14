import logging
import lseg.data as ld
import os
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.contrib.concurrent import process_map
from functools import partial
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import linregress
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

class RunConfig:
    def __init__(self, universe: str, endDate: str, riskFreeRate2Y: float, riskFreeRate5Y: float):
        self.universe = universe
        self.endDate = endDate
        self.riskFreeRate2Y = riskFreeRate2Y
        self.riskFreeRate5Y = riskFreeRate5Y
        self.instruments_to_clean = []

    def add_instruments_to_clean(self, new_instruments):
        for inst in new_instruments:
            if inst not in self.instruments_to_clean:
                self.instruments_to_clean.append(inst)


def refinitiv_session():
    logging.info("Connect to Refnitiv")
    config = ld.get_config()
    config.set_param("logs.transports.console.enabled", True)
    config.set_param("logs.level", "info")
    try:
        ld.open_session()
    except Exception as ex:
        logging.error(ex)
        logging.error("Failed to connect to Refnitiv")
        exit(1)

def get_universe(config: RunConfig):
    logging.info("Get Tickers from Index")
    try:
        query = ld.get_data(
        universe=config.universe,
        fields=['TR.RIC'],
        parameters={'SDate': config.endDate        }
        )
        del query['RIC']
        return set(query['Instrument'].dropna().to_list()), query['Instrument'].dropna()
    except Exception as ex:
        logging.error(ex)
        logging.error("Failed to get Tickers from Index")
        exit(1)

def get_data(config: RunConfig, universe: set, frq: str):
    logging.info("Get Data from Refinitiv")

    date = datetime.strptime(config.endDate, "%Y-%m-%d")
    if frq == "W":
        new_date = date - relativedelta(years=2) + relativedelta(days=1)
        startDate = new_date.strftime("%Y-%m-%d")
    elif frq == "M":
        new_date = date - relativedelta(years=5) + relativedelta(days=1)
        startDate = new_date.strftime("%Y-%m-%d")
    else:
        logging.error("Invalid frq")
        exit(1)

    try:
        prices = ld.get_data(
            universe=universe,
            fields=["TR.PriceClose","TR.PriceClose.date"],
            parameters={
                "Frq": frq,
                "SDate": startDate,
                "EDate": config.endDate
            }
        )
        esg = ld.get_data(
            universe=universe,
            fields=["TR.TRESGScore", "TR.TRESGScore.date"],
            parameters={
                "SDate": startDate
            }
        )
        prices.columns = ["Instrument", "Price Close", "Date"]
        esg.columns = ["Instrument", "ESG Score", "Date"]
    except Exception as ex:
        logging.error(ex)
        logging.error("Failed to get Data from Refinitiv")
        exit(1)

    mask_na = esg["Date"].isna() | esg["ESG Score"].isna()
    instruments_dropped = esg.loc[mask_na, "Instrument"].unique()
    config.add_instruments_to_clean(instruments_dropped)

    esg_dropped = esg.dropna().copy()
    esg_dropped["Date"] = pd.to_datetime(esg_dropped["Date"])
    latest_esg = (
        esg_dropped
        .loc[esg_dropped.groupby("Instrument")["Date"].idxmax()]
        .sort_values("Instrument")
        .reset_index(drop=True)
    )

    prices_clean = prices.dropna(subset=["Price Close", "Date"])
    prices_clean2 = prices_clean.dropna().groupby(["Date", "Instrument"]).mean().reset_index()
    prices_pivot = prices_clean2.dropna().pivot(index="Date", columns="Instrument", values="Price Close")
    return prices_pivot, latest_esg

def clean_data(config: RunConfig, prices5Y, esg5Y, prices2Y, esg2Y):
    #TODO tbd, ob auch Top 10 Returns/Verluste rausgenommen werden sollten

   # 1. Alle Ticker entfernen, die NA Werte beinhalten
    #prices2Y_filtered = prices2Y.drop(
    #    columns=[c for c in prices2Y.columns if c in config.instruments_to_clean])
    #prices5Y_filtered = prices5Y.drop(
    #    columns=[c for c in prices2Y.columns if c in config.instruments_to_clean])
    to_clean5Y = config.instruments_to_clean
    instruments_esg = set(esg5Y["Instrument"])
    instruments_prices = set(prices5Y.columns)
    to_clean5Y.extend(list(instruments_esg.symmetric_difference(instruments_prices)))
    to_clean5Y.extend(list(instruments_prices.symmetric_difference(instruments_esg)))

    to_clean2Y = config.instruments_to_clean
    instruments_esg = set(esg2Y["Instrument"])
    instruments_prices = set(prices2Y.columns)
    to_clean2Y.extend(list(instruments_esg.symmetric_difference(instruments_prices)))
    to_clean2Y.extend(list(instruments_prices.symmetric_difference(instruments_esg)))

    prices2Y_filtered = prices2Y.drop(columns=to_clean2Y)
    prices5Y_filtered = prices5Y.drop(columns=to_clean5Y)

    esg2Y_filtered = esg2Y.loc[
        ~esg2Y["Instrument"].isin(to_clean2Y)
    ].reset_index(drop=True)
    esg5Y_filtered = esg5Y.loc[
        ~esg2Y["Instrument"].isin(to_clean5Y)
    ].reset_index(drop=True)

    esg_inst = set(esg5Y_filtered["Instrument"])
    pri_inst = set(prices5Y_filtered.columns)
    new = list(esg_inst.symmetric_difference(pri_inst))
    new.extend(pri_inst.symmetric_difference(esg_inst))
    prices5Y_filtered2 = prices5Y_filtered.drop(columns=new, errors="ignore")
    esg5Y_filtered2 = esg5Y_filtered.loc[
        ~esg5Y_filtered["Instrument"].isin(new)
    ].reset_index(drop=True)

    return prices5Y_filtered2, esg5Y_filtered2, prices2Y_filtered, esg2Y_filtered

def expected_returns(prices: pd.DataFrame, freq: str):
    # Berechnung der wöchentlichen/monatlichen durchschnittlichen Returns
    # inklusive einer Hochrechnung auf einen jährlichen Return

    if freq == "W":
        scale = 52
    elif freq == "M":
        scale = 12
    else:
        logging.error("Invalid freq")
        exit(1)

    na_counts = prices.isna().sum()
    returns = prices.pct_change(fill_method=None)
    mu = returns.mean() * scale

    return mu

def cov_matrix(prices: pd.DataFrame, freq: str):
    if freq == "W":
        scale = 52
    elif freq == "M":
        scale = 12
    else:
        logging.error("Invalid freq")
        exit(1)

    returns = prices.pct_change(fill_method=None)
    cov = returns.cov() * scale

    return cov


def efficient_frontiers(prices: pd.DataFrame, esg: pd.DataFrame, freq: str, portfolios):
    frontiers = {}
    thresholds=[85,70,55]

    for t in thresholds:
        prices_filtered, esg_filtered = filter_universe(prices, esg, t)
        returns = expected_returns(prices_filtered, freq)
        cov = cov_matrix(prices_filtered, freq)
        frontier = compute_efficient_frontier(returns, cov, portfolios, t)
        frontiers[f"ESG <= {t}"] = frontier

    return frontiers


def filter_universe(prices: pd.DataFrame, esg: pd.DataFrame, t: int):
    instruments_above_X = esg[esg['ESG Score'] > t]['Instrument']
    instruments_list = instruments_above_X.tolist()
    prices_filtered = prices.drop(
        columns=[c for c in prices.columns if c in instruments_list])
    esg_filtered = esg.loc[
        ~esg["Instrument"].isin(instruments_list)
    ].reset_index(drop=True)

    return prices_filtered, esg_filtered


def compute_efficient_frontier(mu, cov_matrix, portfolios, t, max_workers=10):
    mu = np.array(mu)
    cov_matrix = np.array(cov_matrix)
    n_assets = len(mu)

    # Portfolio-Risiko-Funktion
    def portfolio_risk(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)

    # Constraints: Summe der Gewichte = 1, Portfolio-Return = target_return
    def get_constraints(target_return):
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Summe=1
            {'type': 'eq', 'fun': lambda w: w @ mu - target_return}  # Zielrendite
        ]
        return constraints

    bounds = [(0, 0.5) for _ in range(n_assets)]
    target_returns = np.linspace(mu.min(), mu.max(), portfolios)
    x0 = np.repeat(1 / n_assets, n_assets)

    frontier_points = []

    # Optimierung für ein einzelnes Ziel
    def optimize_target(r_target):
        cons = get_constraints(r_target)
        res = minimize(portfolio_risk, x0, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            return {
                'Return': float(r_target),
                'Risk': float(portfolio_risk(res.x)),
                'Weights': res.x
            }
        else:
            logging.error(f"Optimization failed for target return {r_target:.4f}")
            return None

    # Multithreading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(optimize_target, r): r for r in target_returns}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Calculating Efficient Frontier for threshold {t}"):
            result = f.result()
            if result is not None:
                frontier_points.append(result)

    frontier_df = pd.DataFrame(frontier_points).sort_values(by='Return').reset_index(drop=True)

    return frontier_df


def plot_frontiers(frontiers):
    grid_color = (236 / 255, 237 / 255, 239 / 255)
    frontier_color = (0 / 255, 39 / 255, 80 / 255)
    plt.figure(figsize=(12, 8))

    linestyles = [":", "--", "-"]
    for i, ((esg_border, frontier_df)) in enumerate(sorted(frontiers.items())):
        plt.plot(
            frontier_df["Risk"],
            frontier_df["Return"],
            label=esg_border,
            color=frontier_color,
            linestyle=linestyles[i],
            linewidth=2
        )

    plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    plt.grid(True, color=grid_color)

    plt.xlabel("Risk")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)

    plt.show()
    # Plot speichern
    #output_path = outputfile + "/" + name + "/" + "frontiers.png"
    #plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Speicher freigeben, falls du viele Plots erzeugst


def capital_market_line(config: RunConfig, frontiers: dict, freq: str):
    grid_color = (236 / 255, 237 / 255, 239 / 255)
    frontier_color = (0 / 255, 39 / 255, 80 / 255)
    cml_color = (245 / 255, 158 / 255, 0 / 255)
    linestyles = ["-", "--", ":"]
    type = 0

    #rf = config.riskFreeRate
    if freq == "W":
        rf = config.riskFreeRate2Y
    elif freq == "M":
        rf = config.riskFreeRate5Y
    else:
        logging.error("Invalid freq")
        exit(1)

    rf = 0.03               # TODO einmal neu durchlaufen lassen, so dass config geladen wird

    output = {}
    plt.figure(figsize=(12, 8))

    for name, df in frontiers.items():
        returns = df["Return"].values
        risk = df["Risk"].values

        # ---- Sharpe Ratio ----
        sharpe = (returns - rf) / risk

        df["Sharpe"] = sharpe  # optional speichern

        # Tangency Portfolio
        max_idx = np.argmax(sharpe)

        tangency_return = returns[max_idx]
        tangency_risk = risk[max_idx]
        max_sharpe = sharpe[max_idx]
        output[name] = {(max_sharpe, tangency_return, tangency_risk)}

        # ---- Frontier plot ----
        plt.plot(risk, returns, label=f"{name} Frontier", color=frontier_color, linestyle=linestyles[type])

        # ---- Capital Market Line ----
        sigma_range = np.linspace(0, max(risk) * 1.2, 100)
        cml = rf + max_sharpe * sigma_range

        plt.plot(sigma_range, cml, linestyle=linestyles[type], label=f"{name} CML", color=cml_color)
        type = type + 1

    # Risk-free Punkt
    plt.scatter(0, rf, marker="x", s=100, color=cml_color)

    plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    plt.grid(True, color=grid_color)
    plt.xlabel("Risk")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    return pd.DataFrame.from_dict(output, orient="index")


def main():
    #config = RunConfig(universe="0#.SPX", endDate="2025-12-31")
    config = RunConfig(universe="0#.STOXX", endDate="2025-12-31", riskFreeRate2Y=0.02062, riskFreeRate5Y=0.02350)

    # Flag: True = gespeicherte DataFrames + config laden, False = neu abrufen
    use_saved_data = True

    data_folder = "DataFrame"
    os.makedirs(data_folder, exist_ok=True)

    if use_saved_data:
        # Lade alle DataFrames + config
        with open(os.path.join(data_folder, "config.pkl"), "rb") as f:
            config = pickle.load(f)
        with open(os.path.join(data_folder, "prices2Y.pkl"), "rb") as f:
            prices2Y = pickle.load(f)
        with open(os.path.join(data_folder, "esg2Y.pkl"), "rb") as f:
            esg2Y = pickle.load(f)
        with open(os.path.join(data_folder, "prices5Y.pkl"), "rb") as f:
            prices5Y = pickle.load(f)
        with open(os.path.join(data_folder, "esg5Y.pkl"), "rb") as f:
            esg5Y = pickle.load(f)
        with open(os.path.join(data_folder, "df_universe.pkl"), "rb") as f:
            df_universe = pickle.load(f)
    else:
        refinitiv_session()
        universe, df_universe = get_universe(config)
        prices2Y, esg2Y = get_data(config, universe, "W")
        prices5Y, esg5Y = get_data(config, universe, "M")

        # Speichern aller Objekte
        with open(os.path.join(data_folder, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        with open(os.path.join(data_folder, "prices2Y.pkl"), "wb") as f:
            pickle.dump(prices2Y, f)
        with open(os.path.join(data_folder, "esg2Y.pkl"), "wb") as f:
            pickle.dump(esg2Y, f)
        with open(os.path.join(data_folder, "prices5Y.pkl"), "wb") as f:
            pickle.dump(prices5Y, f)
        with open(os.path.join(data_folder, "esg5Y.pkl"), "wb") as f:
            pickle.dump(esg5Y, f)
        with open(os.path.join(data_folder, "df_universe.pkl"), "wb") as f:
            pickle.dump(df_universe, f)

    clean_prices5Y, clean_esg5Y, clean_prices2Y, clean_esg2Y = clean_data(config, prices5Y, esg5Y, prices2Y, esg2Y)
    returns2Y = expected_returns(clean_prices2Y, freq="W")
    returns5Y = expected_returns(clean_prices5Y, freq="M")
    cov_matrix2Y = cov_matrix(clean_prices2Y, freq="W")
    cov_matrix5Y = cov_matrix(clean_prices5Y, freq="M")

    frontiers2Y = efficient_frontiers(clean_prices2Y, clean_esg2Y, "W", portfolios=100)
    plot_frontiers(frontiers2Y)
    cml_output2Y = capital_market_line(config, frontiers2Y, "W")
    frontiers5Y = efficient_frontiers(clean_prices5Y, clean_esg5Y, "M", portfolios=100)
    plot_frontiers(frontiers5Y)
    cml_output5Y = capital_market_line(config, frontiers5Y, "M")


    logging.info("Save data to csv")
    dataframes_to_save = [
        ('01-Universe', df_universe),
        ('02-Prices5Y', prices5Y),
        ('03-ESG5Y', esg5Y),
        ('04-Prices2Y', prices2Y),
        ('05-ESG2Y', esg2Y),
        ('06-Prices5Y_cleaned', clean_prices5Y),
        ('07-ESG5Y_cleaned', clean_esg5Y),
        ('08-Prices2Y_cleaned', clean_prices2Y),
        ('09-ESG2Y_cleaned', clean_esg2Y),
        ('10-Returns5Y', returns5Y),
        ('11-Returns2Y', returns2Y),
        ('12-CovMatrix5Y', cov_matrix5Y),
        ('13-CovMatrix2Y', cov_matrix2Y),
        ('14-Frontiers2Y', frontiers2Y),
        ('15-Frontiers5Y', frontiers5Y),
        ('16-Sharpe2Y', cml_output2Y),
        ('17-Sharpe5Y', cml_output5Y),
    ]

    # Speichern als CSV
    for name, df in dataframes_to_save:
        df.to_csv(f'Data/{name}.csv', index=True)
    ld.close_session()

if __name__ == '__main__':
    pd.set_option('future.no_silent_downcasting', True)
    logging.basicConfig(
        level=logging.INFO,
        filename= 'runtime.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s'
    )
    main()