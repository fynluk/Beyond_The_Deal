import logging
import lseg.data as ld
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

def main():
    config = RunConfig(universe="0#.SPX", endDate="2025-12-31")
    refinitiv_session()
    universe, df_universe = get_universe(config)
    prices5Y, esg5Y = get_data(config, universe, "M")
    prices2Y, esg2Y = get_data(config, universe, "W")
    print(config.instruments_to_clean)

    logging.info("Save data to csv")
    dataframes_to_save = [
        ('01-Universe', df_universe),
        ('02-Prices5Y', prices5Y),
        ('03-ESG5Y', esg5Y),
        ('04-Prices2Y', prices2Y),
        ('05-ESG2Y', esg2Y),
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