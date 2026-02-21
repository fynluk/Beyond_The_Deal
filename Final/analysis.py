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
from concurrent.futures import ProcessPoolExecutor
import statsmodels.api as sm
import statsmodels.graphics.regressionplots as smg
from scipy.stats import norm, jarque_bera

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
        ~esg5Y["Instrument"].isin(to_clean5Y)
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


def plot_esg_distribution(esg_df, freq):
    grid_color = (236 / 255, 237 / 255, 239 / 255)

    esg_values = esg_df["ESG Score"]

    # 5er-Bins von 0 bis 100
    bins = np.arange(0, 105, 5)

    # Farbe (0,39,80)
    base_color = (0/255, 39/255, 80/255)

    plt.figure(figsize=(12, 8))
    plt.gca().set_axisbelow(True)

    plt.hist(
        esg_values,
        bins=bins,
        color=base_color,
        edgecolor="black"
    )

    # Durchschnitt berechnen
    mean = esg_values.mean()

    # Durchschnittslinie einzeichnen
    plt.axvline(mean,
                linestyle="--",
                linewidth=2,
                label=f"Mean = {mean: .1f}")

    plt.xticks(np.arange(0, 105, 5), fontsize=14)
    plt.ylim(0, 120)
    plt.yticks(np.arange(0, 121, 20), fontsize=14)
    plt.xlabel("ESG Score", fontsize=16)
    plt.ylabel("Number of Assets", fontsize=16)
    plt.grid(True, color=grid_color)

    plt.legend(loc="upper left", fontsize=14)
    plt.grid(True)
    if freq == "W":
        plt.savefig("Plots/01-Distribution2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/02-Distribution5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)
    plt.show()
    plt.close()


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

def plot_return_distribution(returns, bins, freq):
    grid_color = (236 / 255, 237 / 255, 239 / 255)
    bar_color = (0 / 255, 39 / 255, 80 / 255)
    norm_color = (245 / 255, 158 / 255, 0 / 255)

    plt.figure(figsize=(12, 8))
    plt.gca().set_axisbelow(True)

    # Histogramm
    plt.hist(returns, bins=bins, density=True, alpha=1, label="Return Distribution", color=bar_color)

    # Mittelwert
    mean = returns.mean()
    plt.axvline(mean, linestyle="--", linewidth=2, label=f"Mean = {mean:.2%}")

    # Normalverteilung
    x = np.linspace(returns.min(), returns.max(), 500)
    plt.plot(x, norm.pdf(x, mean, returns.std()), linewidth=2, label="Normal Distribution", color=norm_color)

    # -------- Normalitätstest --------
    jb_stat, jb_pvalue = jarque_bera(returns)

    if jb_pvalue < 0.05:
        normal_text = "Not normally distributed"
    else:
        normal_text = "Cannot reject normality"

    # Textbox im Plot
    plt.text(0.98, 0.95,
             f"Jarque-Bera p-value = {jb_pvalue:.2e}\n{normal_text}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=12)

    # Titel & Labels
    plt.xlabel("Return", fontsize=16)
    plt.ylabel("Density", fontsize=16)

    # Tick-Größe
    plt.xticks(np.arange(-1.0, 1.01, 0.2), fontsize=14)
    plt.ylim(0, 4)
    plt.yticks(np.arange(0, 4, 0.5), fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Legende oben links
    plt.legend(loc="upper left", fontsize=14)

    plt.grid(True, color=grid_color)
    plt.tight_layout()
    if freq == "W":
        plt.savefig("Plots/03-Distribution2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/04-Distribution5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)
    plt.show()
    plt.close()


def efficient_frontiers(prices: pd.DataFrame, esg: pd.DataFrame, freq: str, portfolios: int):
    logging.info("Calculating efficient frontiers")
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
    frontier_points = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(optimize_target, r_target) for r_target in target_returns]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc=f"Calculating Efficient Frontier for threshold {t}"):
            result = f.result()
            if result is not None:
                frontier_points.append(result)

    frontier_df = pd.DataFrame(frontier_points).sort_values(by='Return').reset_index(drop=True)

    return frontier_df


def plot_frontiers(frontiers, freq):
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
    plt.xlim(0, 0.4)
    plt.ylim(-0.5, 1.0)
    plt.xticks(np.arange(0, 0.41, 0.1))
    plt.yticks(np.arange(-0.5, 1.01, 0.25))

    plt.grid(True, color=grid_color)

    plt.xlabel("Risk", fontsize=16)
    plt.ylabel("Expected Return", fontsize=16)
    plt.legend(loc="upper left", fontsize=14)
    plt.grid(True)

    if freq == "W":
        plt.savefig("Plots/05-EffFrontier2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/06-EffFrontier5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)
    plt.show()
    plt.close()


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
    #plt.scatter(0, rf, marker="x", s=100, color=cml_color)

    plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.xlim(0, 0.4)
    plt.ylim(-0.5, 1.0)
    plt.xticks(np.arange(0, 0.41, 0.1))
    plt.yticks(np.arange(-0.5, 1.01, 0.25))

    # ---- Risk-free Rate als zusätzlicher Y-Tick ----
    current_yticks = plt.gca().get_yticks()

    # Falls rf noch nicht als Tick existiert → hinzufügen
    if not np.isclose(current_yticks, rf).any():
        new_yticks = np.append(current_yticks, rf)
    else:
        new_yticks = current_yticks

    plt.yticks(new_yticks)

    # Ticklabels einfärben
    for tick, label in zip(plt.gca().get_yticks(), plt.gca().get_yticklabels()):
        if np.isclose(tick, rf):
            label.set_color(cml_color)
            label.set_fontweight("bold")

    plt.grid(True, color=grid_color)
    plt.xlabel("Risk", fontsize=16)
    plt.ylabel("Expected Return", fontsize=16)
    plt.legend(loc="upper left", fontsize=14)
    plt.grid(True)
    if freq == "W":
        plt.savefig("Plots/07-CML2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/08-CML5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)
    plt.show()
    plt.close()

    return pd.DataFrame.from_dict(output, orient="index")


def monte_carlo_portfolio(returns: pd.DataFrame, esg: pd.DataFrame, cov_matrix: pd.DataFrame, portfolios: int, seed: int):
    np.random.seed(seed)

    # Gemeinsame Instrumente
    common_assets = returns.index.intersection(esg["Instrument"])

    returns = returns.loc[common_assets].values
    esg = (
        esg.set_index("Instrument")
        .loc[common_assets]["ESG Score"]
        .values
    )

    cov_matrix = cov_matrix.loc[common_assets, common_assets].values

    n_assets = len(common_assets)

    portfolio_results = []

    for _ in range(portfolios):
        # Zufällige Gewichte (Long-only)
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        # Portfolio Return
        port_return = np.dot(weights, returns)

        # Portfolio Risiko (Std Dev)
        port_risk = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )

        # Gewichteter ESG Score
        port_esg = np.dot(weights, esg)

        portfolio_results.append(
            [port_return, port_esg, port_risk, weights]
        )

    portfolio_df = pd.DataFrame(
        portfolio_results,
        columns=["Return", "ESG Score", "Risk", "Weights"]
    )

    return portfolio_df


def multiregression(portfolios: pd.DataFrame):
    # Dependent Variable
    y = portfolios["Return"]

    # Independent Variables
    X = portfolios[["ESG Score", "Risk"]]

    # Konstante hinzufügen
    X = sm.add_constant(X)

    # OLS
    model = sm.OLS(y, X)

    results = model.fit(cov_type="HC3")  # White-robust

    return results


def plot_regression_summary(model, freq):
    summary_text = model.summary().as_text()

    fig = plt.figure(figsize=(12, 10))
    plt.text(0.01, 0.99, summary_text,
             fontsize=18,
             verticalalignment='top',
             family='monospace')

    plt.axis('off')
    plt.tight_layout()
    if freq == "W":
        plt.savefig("Plots/09-Regression2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/10-Regression5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)
    plt.show()
    plt.close()


def plot_partial_regression(model, freq):
    grid_color = (236 / 255, 237 / 255, 239 / 255)
    point_color = (0 / 255, 39 / 255, 80 / 255)
    line_color = (245 / 255, 158 / 255, 0 / 255)

    y = model.model.endog
    X = model.model.exog
    var_names = model.model.exog_names

    # Indizes bestimmen
    idx_esg = var_names.index("ESG Score")
    idx_risk = var_names.index("Risk")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, idx, name in zip(
        axes,
        [idx_esg, idx_risk],
        ["ESG Score", "Risk"]
    ):

        # Residuen von Y auf andere Variablen
        other_idx = [i for i in range(X.shape[1]) if i != idx and var_names[i] != "const"]

        X_other = X[:, other_idx]
        X_target = X[:, idx]

        # Y ~ andere Variablen
        beta_y = np.linalg.lstsq(X_other, y, rcond=None)[0]
        resid_y = y - X_other @ beta_y

        # X_target ~ andere Variablen
        beta_x = np.linalg.lstsq(X_other, X_target, rcond=None)[0]
        resid_x = X_target - X_other @ beta_x

        # Scatter
        ax.scatter(resid_x, resid_y, alpha=0.4, color=point_color)

        # Regressionslinie
        slope, intercept = np.polyfit(resid_x, resid_y, 1)
        x_vals = np.linspace(resid_x.min(), resid_x.max(), 100)
        ax.plot(x_vals, slope * x_vals + intercept,
                color=line_color,
                linewidth=2)

        # Slope im Plot anzeigen
        ax.text(
            0.05, 0.90,
            f"Slope = {slope:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top"
        )

        ax.set_xlabel(f"{name} (partialled out)", fontsize=16)
        ax.set_ylabel("Return (partialled out)", fontsize=16)

        ax.grid(True, color=grid_color)
        ax.tick_params(labelsize=14)

    plt.tight_layout()

    if freq == "W":
        plt.savefig("Plots/11-PartialRegression2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/12-PartialRegression5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)

    plt.show()
    plt.close()


def plot_regression_surface_3d(model, portfolios, freq):
    grid_color = (236 / 255, 237 / 255, 239 / 255)
    surface_color = (0 / 255, 39 / 255, 80 / 255)

    # Regressionskoeffizienten
    beta_0 = model.params["const"]
    beta_esg = model.params["ESG Score"]
    beta_risk = model.params["Risk"]

    # Wertebereiche definieren
    esg_vals = np.linspace(portfolios["ESG Score"].min(),
                           portfolios["ESG Score"].max(), 50)

    risk_vals = np.linspace(portfolios["Risk"].min(),
                            portfolios["Risk"].max(), 50)

    ESG_grid, Risk_grid = np.meshgrid(esg_vals, risk_vals)

    # Regressionsfläche
    Return_grid = (
        beta_0
        + beta_esg * ESG_grid
        + beta_risk * Risk_grid
    )

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        ESG_grid,
        Risk_grid,
        Return_grid,
        color=surface_color,
        alpha=0.9,
        edgecolor='none'
    )

    # ---------- Labels ----------
    ax.set_xlabel("ESG Score", fontsize=16, labelpad=20)
    ax.set_ylabel("Risk", fontsize=16, labelpad=20)
    ax.invert_yaxis()
    ax.set_zlabel("Expected Return", fontsize=16, labelpad=25)

    # ---------- Prozentformat ----------
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.zaxis.set_major_formatter(PercentFormatter(1, decimals=1))

    # ---------- Tick Größe ----------
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # ---------- Grid Styling ----------
    ax.xaxis._axinfo["grid"]["color"] = grid_color
    ax.yaxis._axinfo["grid"]["color"] = grid_color
    ax.zaxis._axinfo["grid"]["color"] = grid_color

    # Perspektive
    ax.view_init(elev=25, azim=135)

    # Wichtig bei 3D (statt tight_layout)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    if freq == "W":
        plt.savefig("Plots/13-RegressionSurface2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/14-RegressionSurface5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)

    plt.show()
    plt.close()


def plot_mc_risk_return(portfolios: pd.DataFrame, freq: str):
    grid_color = (236 / 255, 237 / 255, 239 / 255)
    point_color = (0 / 255, 39 / 255, 80 / 255)
    line_color = (245 / 255, 158 / 255, 0 / 255)

    plt.figure(figsize=(12, 8))
    plt.gca().set_axisbelow(True)

    x = portfolios["Risk"].values
    y = portfolios["Return"].values

    # Scatter
    plt.scatter(
        x,
        y,
        alpha=0.3,
        s=5,
        color=point_color
    )

    # ----- Regression -----
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    # R² berechnen
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    plt.plot(
        x_line,
        y_line,
        color=line_color,
        linewidth=2
    )

    # Textbox
    plt.text(
        0.05, 0.95,
        f"Slope = {slope:.4f}\nR² = {r_squared:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=12
    )

    # Formatierung
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    plt.xlim(0.11, 0.15)
    plt.ylim(0, 0.2)

    plt.xticks(np.arange(0.11, 0.15, 0.01), fontsize=14)
    plt.yticks(np.arange(0, 0.21, 0.05), fontsize=14)

    plt.xlabel("Risk", fontsize=16)
    plt.ylabel("Expected Return", fontsize=16)

    plt.grid(True, color=grid_color)

    if freq == "W":
        plt.savefig("Plots/15-MC-RiskReturn-2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/16-MC-RiskReturn-5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)

    plt.show()
    plt.close()


def plot_mc_esg_return(portfolios: pd.DataFrame, freq: str):
    grid_color = (236 / 255, 237 / 255, 239 / 255)
    point_color = (0 / 255, 39 / 255, 80 / 255)
    line_color = (245 / 255, 158 / 255, 0 / 255)

    plt.figure(figsize=(12, 8))
    plt.gca().set_axisbelow(True)

    x = portfolios["ESG Score"].values
    y = portfolios["Return"].values

    # Scatter
    plt.scatter(
        x,
        y,
        alpha=0.3,
        s=5,
        color=point_color
    )

    # ----- Regression -----
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept

    # R² berechnen
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    plt.plot(
        x_line,
        y_line,
        color=line_color,
        linewidth=2
    )

    # Textbox
    plt.text(
        0.05, 0.95,
        f"Slope = {slope:.6f}\nR² = {r_squared:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        fontsize=12
    )

    # Formatierung
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

    plt.ylim(0, 0.20)
    plt.yticks(np.arange(0, 0.21, 0.05), fontsize=14)

    plt.xlim(66, 73)
    plt.xticks(np.arange(66, 73, 1), fontsize=14)

    plt.xlabel("ESG Score", fontsize=16)
    plt.ylabel("Expected Return", fontsize=16)

    plt.grid(True, color=grid_color)

    if freq == "W":
        plt.savefig("Plots/17-MC-ESGReturn-2Y.png", dpi=300, bbox_inches='tight')
    elif freq == "M":
        plt.savefig("Plots/18-MC-ESGReturn-5Y.png", dpi=300, bbox_inches='tight')
    else:
        logging.error("Invalid freq")
        exit(1)

    plt.show()
    plt.close()


def main():
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
    plot_esg_distribution(clean_esg2Y, "W")
    plot_esg_distribution(clean_esg5Y, "M")
    returns2Y = expected_returns(clean_prices2Y, freq="W")
    returns5Y = expected_returns(clean_prices5Y, freq="M")
    plot_return_distribution(returns2Y, 40, "W")
    plot_return_distribution(returns5Y, 40, "M")
    cov_matrix2Y = cov_matrix(clean_prices2Y, freq="W")
    cov_matrix5Y = cov_matrix(clean_prices5Y, freq="M")

    if use_saved_data:
        with open(os.path.join(data_folder, "frontiers2Y.pkl"), "rb") as f:
            frontiers2Y = pickle.load(f)
        with open(os.path.join(data_folder, "frontiers5Y.pkl"), "rb") as f:
            frontiers5Y = pickle.load(f)
    else:
        frontiers2Y = efficient_frontiers(clean_prices2Y, clean_esg2Y, "W", portfolios=100)
        frontiers5Y = efficient_frontiers(clean_prices5Y, clean_esg5Y, "M", portfolios=100)

        with open(os.path.join(data_folder, "frontiers2Y.pkl"), "wb") as f:
            pickle.dump(frontiers2Y, f)
        with open(os.path.join(data_folder, "frontiers5Y.pkl"), "wb") as f:
            pickle.dump(frontiers5Y, f)

    plot_frontiers(frontiers2Y, "W")
    plot_frontiers(frontiers5Y, "M")
    cml_output2Y = capital_market_line(config, frontiers2Y, "W")
    cml_output5Y = capital_market_line(config, frontiers5Y, "M")

    MCportfolios2Y = monte_carlo_portfolio(returns2Y, clean_esg2Y, cov_matrix2Y, 100000, 45768)
    MCportfolios5Y = monte_carlo_portfolio(returns5Y, clean_esg5Y, cov_matrix5Y, 100000, 44227)
    regression2Y = multiregression(MCportfolios2Y)
    regression5Y = multiregression(MCportfolios5Y)
    plot_regression_summary(regression2Y, "W")
    plot_regression_summary(regression5Y, "M")
    plot_partial_regression(regression2Y, "W")
    plot_partial_regression(regression5Y, "M")
    plot_regression_surface_3d(regression2Y, MCportfolios2Y, "W")
    plot_regression_surface_3d(regression5Y, MCportfolios5Y, "M")
    plot_mc_risk_return(MCportfolios2Y, "W")
    plot_mc_risk_return(MCportfolios5Y, "M")
    plot_mc_esg_return(MCportfolios2Y, "W")
    plot_mc_esg_return(MCportfolios5Y, "M")


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
        ('14-Frontiers2Y', pd.concat(frontiers2Y, names=["Frontier"])),
        ('15-Frontiers5Y', pd.concat(frontiers5Y, names=["Frontier"])),
        ('16-Sharpe2Y', cml_output2Y),
        ('17-Sharpe5Y', cml_output5Y),
        ('18-MC-Portfolios2Y', MCportfolios2Y),
        ('19-MC-Portfolios5Y', MCportfolios5Y)
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