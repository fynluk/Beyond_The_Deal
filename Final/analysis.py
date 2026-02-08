import logging
import lseg.data as ld
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
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
    def __init__(self, universe: str, endDate: str):
        self.universe = universe
        self.endDate = endDate
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
    prices_pivot = prices.dropna().pivot(index="Date", columns="Instrument", values="Price Close")
    return prices_pivot, latest_esg

def clean_data(config: RunConfig, prices5Y, esg5Y, prices2Y, esg2Y):
    #TODO tbd, ob auch Top 10 Returns/Verluste rausgenommen werden sollten

   # 1. Alle Ticker entfernen, die NA Werte beinhalten
    #prices2Y_filtered = prices2Y.drop(
    #    columns=[c for c in prices2Y.columns if c in config.instruments_to_clean])
    #prices5Y_filtered = prices5Y.drop(
    #    columns=[c for c in prices2Y.columns if c in config.instruments_to_clean])
    prices2Y_filtered = prices2Y.drop(columns=config.instruments_to_clean)
    prices5Y_filtered = prices5Y.drop(columns=config.instruments_to_clean)

    display(esg2Y)
    esg2Y_filtered = esg2Y.loc[
        ~esg2Y["Instrument"].isin(config.instruments_to_clean)
    ].reset_index(drop=True)
    display(len(config.instruments_to_clean))
    display(esg2Y_filtered)
    esg5Y_filtered = esg5Y.loc[
        ~esg2Y["Instrument"].isin(config.instruments_to_clean)
    ].reset_index(drop=True)

    return prices5Y_filtered, esg5Y_filtered, prices2Y_filtered, esg2Y_filtered

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

    returns = prices.pct_change()
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

    returns = prices.pct_change()
    cov = returns.cov() * scale

    return cov


def efficient_frontiers(prices: pd.DataFrame, esg: pd.DataFrame, freq: str):
    frontiers = {}
    thresholds=[25,50,75]

    for t in thresholds:
        prices_filtered, esg_filtered = filter_universe(prices, esg, t)
        returns = expected_returns(prices_filtered, freq)
        cov = cov_matrix(prices_filtered, freq)

        exit(1)


def filter_universe(prices: pd.DataFrame, esg: pd.DataFrame, t: int):
    instruments_below_X = esg[esg['ESG Score'] < t]['Instrument']
    instruments_list = instruments_below_X.tolist()
    instruments_list.append("A.N")
    display(instruments_list)
    prices_filtered = prices.drop(
        columns=[c for c in prices.columns if c in instruments_list])
    esg_filtered = esg.loc[
        ~esg["Instrument"].isin(instruments_list)
    ].reset_index(drop=True)
    #display(prices_filtered)
    #display(esg_filtered)

    return prices_filtered, esg_filtered





def main():
    config = RunConfig(universe="0#.SPX", endDate="2025-12-31")
    #config = RunConfig(universe="0#.STOXX", endDate="2025-12-31")  # TODO Bug beheben

    # Flag: True = gespeicherte DataFrames laden, False = neu abrufen
    use_saved_data = False

    data_folder = "DataFrames"
    os.makedirs(data_folder, exist_ok=True)

    if use_saved_data:
        # Lade alle DataFrames
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

        # Speichern
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

    efficient_frontiers(prices2Y, esg2Y, "W")


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
    ]

    # Speichern als CSV
    for name, df in dataframes_to_save:
        df.to_csv(f'Data/{name}.csv', index=True)

if __name__ == '__main__':
    pd.set_option('future.no_silent_downcasting', True)
    logging.basicConfig(
        level=logging.INFO,
        filename= 'runtime.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s'
    )
    main()