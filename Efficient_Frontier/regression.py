import random
import time
from datetime import datetime
import lseg.data as ld
import pickle
import os
import pandas as pd
import logging
import yaml
from matplotlib import pyplot as plt
import concurrent.futures
from tqdm import tqdm
import argparse
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List
from scipy.stats import linregress

def open_ld():
    try:
        ld.open_session()
    except Exception as e:
        logging.error("Refinitive Bridge connection failed; Exception: " + str(e))

def get_data(tickers):
    dict = {}

    def fetch_ticker(t):
        data = ld.get_data(
            universe=t,
            fields=["TR.TRESGScore.date", "TR.TotalReturn1Wk", "TR.TRESGScore", "TR.GovernancePillarScore", "TR.EnvironmentPillarScore", "TR.SocialPillarScore"],
            parameters={
                "Frq": "W",
                "SDate": "2023-01-01",
                "EDate": "2024-12-31"
            }
        )
        latest = data.sort_values("Date").iloc[[-1]]

        time.sleep(4)
        return t, latest

    # ThreadPoolExecutor für parallele Abfragen
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Futures erstellen
        futures = {executor.submit(fetch_ticker, row['Instrument']): row['Instrument'] for index, row in
                   tickers.iterrows()}

        # Fortschrittsanzeige mit tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching tickers"):
            ticker, df = future.result()
            dict[ticker] = df

    return dict

def prepare_regression_data(data_dict: Dict[str, pd.DataFrame],
                            return_col: str = "1 Week Total Return",
                            criteria: List[str] = None,
                            date_col: str = "Date") -> Dict[str, pd.DataFrame]:

    if criteria is None:
        # Default-Kriterien (falls nicht übergeben)
        criteria = ["ESG Score",
                    "Governance Pillar Score",
                    "Environmental Pillar Score",
                    "Social Pillar Score"]

    # Ergebnis-Dict
    result = {crit: [] for crit in criteria}

    for ticker, df in data_dict.items():
        if df is None or df.empty:
            logging.warning("No data found for ticker " + ticker)
            continue

        d = df.copy()

        # Robust: falls Datumsspalte vorhanden, parse; sonst leave as is
        if date_col in d.columns:
            d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
        else:
            d[date_col] = pd.NaT

        # Sicherstellen, dass Return-Spalte numeric
        if return_col in d.columns:
            d[return_col] = pd.to_numeric(d[return_col], errors='coerce')
        else:
            logging.warning("No returns found for ticker " + ticker)
            continue

        # Für jedes Kriterium die Werte extrahieren (coerce numeric)
        for crit in criteria:
            if crit in d.columns:
                values = pd.to_numeric(d[crit], errors='coerce')
                # Baue kleines temp-DF der gültigen Zeilen
                temp = pd.DataFrame({
                    "Ticker": ticker,
                    "Date": d[date_col],
                    "Return": d[return_col],
                    "Value": values
                })
                # Drop rows mit NaN in Return oder Value
                temp = temp.dropna(subset=["Return", "Value"]).reset_index(drop=True)
                if not temp.empty:
                    result[crit].append(temp)
            else:
                # Spalte fehlt: überspringen (kann loggen, wenn gewünscht)
                continue

    # Für jedes Kriterium die Liste von DataFrames zu einem großen DataFrame konkatenieren
    for crit in list(result.keys()):
        if len(result[crit]) == 0:
            result[crit] = pd.DataFrame(columns=["Ticker", "Date", "Return", "Value"])
        else:
            result[crit] = pd.concat(result[crit], ignore_index=True)

    return result


def plot_criteria_scatter(prepared_dict: Dict[str, pd.DataFrame],
                          save_dir: str = None,
                          figsize=(8, 6),
                          marker_size=10,
                          alpha=0.6,
                          annotate_samples: int = 0):
    axes_dict = {}

    for crit, df in prepared_dict.items():
        # Defensive copy + coercion
        if not isinstance(df, pd.DataFrame):
            logging.warning("Plot skipped for Criteria (Type Problem): " + crit)
            continue

        d = df.copy()
        # Ensure required cols exist
        if not set(['Ticker', 'Return', 'Value']).issubset(d.columns):
            logging.warning("Plot skipped for Criteria (Not enough columns): " + crit)
            continue

        # Coerce to numeric and drop NA
        d['Return'] = pd.to_numeric(d['Return'], errors='coerce')
        d['Value'] = pd.to_numeric(d['Value'], errors='coerce')
        d = d.dropna(subset=['Return', 'Value']).reset_index(drop=True)

        if d.empty:
            logging.warning("Plot skipped for Criteria (Not valid data): " + crit)
            continue

        x = d['Value'].values
        y = d['Return'].values

        # Linear regression
        lr = linregress(x, y)
        slope, intercept, r_value, p_value = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
        r2 = r_value ** 2

        # Prepare figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, y, s=marker_size, alpha=alpha, c='#e9a239' , edgecolor='k', linewidth=0.3)
        ax.set_title(f"{crit}\n(n={len(d)})", fontsize=12)
        ax.set_xlabel(crit)
        ax.set_ylabel("Return")

        # Regression line (plot over x-range)
        x_lin = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        y_lin = intercept + slope * x_lin
        ax.plot(x_lin, y_lin, linestyle='-', linewidth=1.5, label='Linear fit', color='#0c264d')

        # Annotate slope, R^2, p-value
        text = f"slope={slope:.4f}\nR²={r2:.3f}\np={p_value:.3g}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

        # Optional: annotate largest residuals (most informative outliers)
        if annotate_samples and annotate_samples > 0:
            preds = intercept + slope * x
            residuals = np.abs(y - preds)
            top_idx = np.argsort(residuals)[-annotate_samples:]
            for i in top_idx:
                ticker = str(d.loc[i, 'Ticker'])
                ax.annotate(ticker, (x[i], y[i]), textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.9)

        plt.tight_layout()

        safe_name = "".join(c if c.isalnum() or c in " _-." else "_" for c in crit)[:200]
        out_path = os.path.join(save_dir, f"{safe_name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info("Plot saved for criteria: " + crit + " to: " + out_path)

def main(folder, tickers):
    logging.info("Open Refinitiv Session for Regression")
    open_ld()

    logging.info("Get Data for Tickers")
    data = get_data(tickers)

    logging.info("Prepare Regression Data")
    regression_data = prepare_regression_data(data)
    print(regression_data)

    logging.info("Plot Data")
    plot_criteria_scatter(regression_data, folder)