from dataclasses import fields

import lseg.data as ld
import pandas
import pickle
import os
import progressbar
import pandas as pd
import yaml
import logging
from IPython.display import display
from matplotlib import pyplot as plt
from refinitiv.dataplatform import search, get_historical_price_summaries
import refinitiv.data as rd
import concurrent.futures
from tqdm import tqdm
import argparse
import numpy as np



def open_ld():
    try:
        ld.open_session()
        rd.open_session()
    except:
        logging.error("Refinitive Bridge connection failed")


def get_tickers():
    #TODO Add more Tickers FTSE100 MSCI ...
    query = ld.get_data(
        universe="0#.SPX",
        fields=["TR.IndexConstituentRIC"],
        parameters={
            'SDate': '2024-12-31'
        }
    )
    return query


def get_historical_price(tickers, max_workers=20):
    """
    Holt historische Kurse für alle Ticker parallel und zeigt Fortschritt an.

    Input:
        tickers: DataFrame mit Spalte 'Instrument' (Ticker-Namen)
        max_workers: Anzahl paralleler Threads
    Output:
        prices_dict: Dictionary, Keys = Ticker, Values = DataFrame mit historischen Preisen
    """
    prices_dict = {}  # Dictionary für Ergebnisse

    # Hilfsfunktion für einzelne Abfrage
    def fetch_ticker(ticker):
        df = ld.get_data(
            universe=ticker,
            fields=["TR.PriceClose","TR.PriceClose.date"],
            parameters={
                "Frq": "D",
                "SDate": "2022-01-01",
                "EDate": "2024-12-31"
            }
        )
        return ticker, df

    # ThreadPoolExecutor für parallele Abfragen
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Futures erstellen
        futures = {executor.submit(fetch_ticker, row['Instrument']): row['Instrument'] for index, row in
                   tickers.iterrows()}

        # Fortschrittsanzeige mit tqdm
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching tickers"):
            ticker, df = future.result()
            prices_dict[ticker] = df

    return prices_dict


def calculate_return_risk(prices_dict, freq='W', max_workers=20):
    """
    Berechnet annualisierte erwartete Rendite und Risiko für jeden Ticker parallel und zeigt Fortschritt an.

    Inputs:
        prices_dict: Dictionary, Keys = Ticker, Values = DataFrame mit Spalten ['Price Close', 'Date']
        freq: 'D' = täglich, 'W' = wöchentlich, 'M' = monatlich
        max_workers: Anzahl paralleler Threads
    Output:
        df_risk_return: DataFrame mit Spalten ['Ticker', 'Return', 'Risk']
    """

    def process_ticker(ticker, df):
        """
        Berechnet Return und Risiko für einen einzelnen Ticker
        """
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Renditeserie erstellen
        if freq == 'D':
            pct_change = df['Price Close'].pct_change().dropna()
            scale = 252
        elif freq == 'W':
            weekly_prices = df.resample('W', on='Date')['Price Close'].last()
            pct_change = weekly_prices.pct_change().dropna()
            scale = 52
        elif freq == 'M':
            monthly_prices = df.resample('M', on='Date')['Price Close'].last()
            pct_change = monthly_prices.pct_change().dropna()
            scale = 12
        else:
            raise ValueError("freq muss 'D', 'W' oder 'M' sein")

        mean_return = pct_change.mean() * scale
        risk = pct_change.std() * np.sqrt(scale)

        return ticker, mean_return, risk

    tickers_list = []
    returns_list = []
    risks_list = []

    # Multithreading mit Fortschrittsanzeige
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_ticker, ticker, df): ticker for ticker, df in prices_dict.items()}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                           desc="Calculating Return & Risk"):
            ticker, mean_return, risk = future.result()
            tickers_list.append(ticker)
            returns_list.append(mean_return)
            risks_list.append(risk)

    # DataFrame erstellen
    df_risk_return = pd.DataFrame({
        'Ticker': tickers_list,
        'Return': returns_list,
        'Risk': risks_list
    })

    return df_risk_return


def plot_return_risk(df_risk_return, title='Return-Risk Scatterplot'):
    """
    Plottet ein Return-Risk Scatterplot für alle Ticker in Prozent.

    Inputs:
        df_risk_return: DataFrame mit Spalten ['Ticker', 'Return', 'Risk']
        title: Titel des Plots
    """
    plt.figure(figsize=(12,8))

    # Scatterplot in Prozent
    plt.scatter(df_risk_return['Risk']*100, df_risk_return['Return']*100, c='blue', alpha=0.6)

    # Ticker als Labels hinzufügen
    for i, row in df_risk_return.iterrows():
        plt.annotate(row['Ticker'], (row['Risk']*100, row['Return']*100), fontsize=8, alpha=0.7)

    plt.xlabel('Risk (%)')
    plt.ylabel('Expected Return (%)')
    plt.title(title)
    plt.grid(True)
    plt.show()


def get_esg_scores(return_risk, max_workers=10):
    """
    Holt den ESG-Score für jeden Ticker und erweitert return_risk DataFrame.

    Inputs:
        return_risk: DataFrame mit Spalten ['Ticker', 'Return', 'Risk']
        max_workers: Anzahl paralleler Threads
    Output:
        return_risk_esg: DataFrame mit zusätzlicher Spalte 'ESG'
    """

    esg_dict = {}  # Dictionary für ESG Scores

    def fetch_esg(ticker):
        try:
            df = ld.get_data(
                universe=ticker,
                fields=["TR.TRESGScore"],
                parameters={'SDate': '2024-12-31'}
            )
            # ESG Score extrahieren; falls mehrere Zeilen, nur erste nehmen
            score = df['ESG Score'].iloc[0] if not df.empty else None
        except:
            score = None
        return ticker, score

    # Multithreading mit Fortschrittsanzeige
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_esg, row['Ticker']): row['Ticker'] for index, row in return_risk.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching ESG Scores"):
            ticker, score = future.result()
            esg_dict[ticker] = score

    # Neue Spalte hinzufügen
    return_risk_esg = return_risk.copy()
    return_risk_esg['ESG'] = return_risk_esg['Ticker'].map(esg_dict)

    return return_risk_esg


def plot_return_risk_esg(df_risk_esg, title='Return-Risk Scatterplot nach ESG', cmap='RdYlGn'):
    """
    Plottet Return-Risk Scatterplot, wobei die Farbe der Punkte den ESG Score zeigt.

    Inputs:
        df_risk_esg: DataFrame mit Spalten ['Ticker', 'Return', 'Risk', 'ESG']
        title: Titel des Plots
        cmap: Colormap für ESG (default: 'RdYlGn', Grün=hoch, Rot=niedrig)
    """
    # ESG-Spalte sicher in numerisch umwandeln
    df_risk_esg['ESG'] = pd.to_numeric(df_risk_esg['ESG'], errors='coerce')

    # Zeilen ohne ESG-Wert entfernen
    df_risk_esg_plot = df_risk_esg.dropna(subset=['ESG']).copy()

    plt.figure(figsize=(12, 8))

    # Scatterplot in Prozent mit ESG als Farbe
    sc = plt.scatter(
        df_risk_esg_plot['Risk'] * 100,
        df_risk_esg_plot['Return'] * 100,
        c=df_risk_esg_plot['ESG'],
        cmap=cmap,
        alpha=0.8,
        edgecolors='k'
    )

    # Ticker als Labels hinzufügen
    for i, row in df_risk_esg_plot.iterrows():
        plt.annotate(row['Ticker'], (row['Risk'] * 100, row['Return'] * 100), fontsize=8, alpha=0.7)

    plt.xlabel('Risk (%)')
    plt.ylabel('Expected Return (%)')
    plt.title(title)
    plt.grid(True)

    # Farbleiste anzeigen
    cbar = plt.colorbar(sc)
    cbar.set_label('ESG Score')

    plt.show()


def main(load_from_file):
    pickle_file = '../Data/historical_prices.pkl'

    logging.info("Open Refinitiv Session")
    open_ld()

    if load_from_file and os.path.exists(pickle_file):
        logging.info(f"Loading historical prices from {pickle_file}")
        with open(pickle_file, 'rb') as f:
            historical_prices = pickle.load(f)
    else:
        logging.info("Get Tickers")
        tickers = get_tickers()
        logging.info("Get historical prices")
        historical_prices = get_historical_price(tickers)
        # Pickle speichern
        with open(pickle_file, 'wb') as f:
            pickle.dump(historical_prices, f)
        logging.info(f"Saved historical prices to {pickle_file}")

    logging.info("Calculate Return Risk")
    return_risk = calculate_return_risk(historical_prices)
    logging.info("Get ESG-Scores")
    return_risk_esg = get_esg_scores(return_risk)
    logging.info("Plot Return Risk ESG")
    plot_return_risk_esg(return_risk_esg)

    ld.close_session()

if __name__ == "__main__":
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        filename="../runtime.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s"
    )

    pd.set_option("future.no_silent_downcasting", True)

    # Kommandozeilenparameter parsen
    parser = argparse.ArgumentParser(description="Run Efficient Frontier Pipeline")
    parser.add_argument(
        '--load_from_file',
        action='store_true',
        help='Load historical prices from pickle instead of fetching from API'
    )
    args = parser.parse_args()

    # main() aufrufen mit Boolean
    main(args.load_from_file)