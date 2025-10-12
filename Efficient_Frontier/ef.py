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


def open_ld():
    try:
        ld.open_session()
    except Exception as e:
        logging.error("Refinitive Bridge connection failed; Exception: " + str(e))


def get_tickers(universe, start):
    #TODO Add more Tickers FTSE100 MSCI ...
    query = ld.get_data(
        universe=universe,
        fields=["TR.IndexConstituentRIC"],
        parameters={
            'SDate': start
        }
    )
    return query.head(1)
    #return query

def get_historical_price(tickers, start, end, max_workers=10):
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
    def fetch_ticker(t):
        data = ld.get_data(
            universe=t,
            fields=["TR.PriceClose","TR.PriceClose.date"],
            parameters={
                "Frq": "D",
                "SDate": start,
                "EDate": end
            }
        )
        time.sleep(3)
        return t, data

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

def calculate_cov_matrix(prices_dict, freq):
    """
    Berechnet annualisierte Kovarianzmatrix der Renditen aus historical_prices.

    Inputs:
      - prices_dict: dict {Ticker: DataFrame mit ['Instrument','Price Close','Date']}
      - freq: 'D'=daily, 'W'=weekly, 'M'=monthly
    Output:
      - cov_matrix: pd.DataFrame, annualisierte Kovarianzmatrix
      - returns_df: pd.DataFrame, annualisierte Renditen pro Ticker
    """
    price_data = pd.DataFrame()

    # Alle Preise in ein DataFrame
    for ticker, df in prices_dict.items():
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        price_data[ticker] = df['Price Close']

    if freq not in ('D', 'W', 'M'):
        raise ValueError("freq muss 'D', 'W' oder 'M' sein")

    # Renditen berechnen
    if freq == 'D':
        returns = price_data.pct_change().dropna()
        scale = 252  # Handelstage
    elif freq == 'W':
        weekly_prices = price_data.resample('W').last()
        returns = weekly_prices.pct_change().dropna()
        scale = 52
    elif freq == 'M':
        monthly_prices = price_data.resample('M').last()
        returns = monthly_prices.pct_change().dropna()
        scale = 12
    else:
        raise ValueError("freq muss 'D', 'W' oder 'M' sein")

    # Annualisierte Renditen
    annual_returns = returns.mean() * scale

    # Annualisierte Kovarianzmatrix
    cov_matrix = returns.cov() * scale

    return cov_matrix, annual_returns


def get_esg_scores(tickers_df, end, max_workers=10):
    """
    Holt den ESG-Score für jeden Ticker in tickers_df und gibt ein DataFrame zurück.

    Inputs:
        tickers_df: DataFrame mit Spalte ['Ticker']
        max_workers: Anzahl paralleler Threads
    Output:
        esg_df: DataFrame mit Spalten ['Ticker', 'ESG']
    """

    results = []

    def fetch_esg(t):
        try:
            df = ld.get_data(
                universe=t,
                fields=["TR.TRESGScore"],
                parameters={'SDate': end}
            )
            s = df['ESG Score'].iloc[0] if not df.empty else None
            time.sleep(3)
        except Exception as e:
            logging.warning("No ESG-Data for Ticker: " + t + " Exception: " + str(e))
            s = None
        return t, s

    # Multithreading mit Fortschrittsanzeige
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_esg, row['Instrument']): row['Instrument'] for _, row in tickers_df.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Fetching ESG Scores"):
            ticker, score = future.result()
            results.append((ticker, score))

    # Ergebnisse in DataFrame umwandeln
    esg_df = pd.DataFrame(results, columns=['Ticker', 'ESG'])

    return esg_df

def markowitz_frontier(mu, cov_matrix, n_points=100, allow_short=False):
    """
    Berechnet die Markowitz-Efficient Frontier mit Progressbar.

    Inputs:
        mu: pd.Series oder np.array, erwartete annualisierte Renditen der Assets
        cov_matrix: pd.DataFrame oder np.array, annualisierte Kovarianzmatrix der Assets
        n_points: int, Anzahl der Zielrenditen für die Frontier
        allow_short: bool, ob Short-Selling erlaubt ist (Gewichte < 0)

    Output:
        frontier_df: DataFrame mit ['Return', 'Risk', 'Weights']
    """

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

    # Bounds für Gewichte
    if allow_short:
        bounds = [(None, None) for _ in range(n_assets)]
    else:
        bounds = [(0,1) for _ in range(n_assets)]

    # Zielrenditen: von minimaler bis maximaler Asset-Return
    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    frontier_points = []

    # Startwerte: gleichverteilte Gewichte
    x0 = np.repeat(1/n_assets, n_assets)

    # Schleife mit Progressbar
    for r_target in tqdm(target_returns, desc="Calculating Efficient Frontier"):
        cons = get_constraints(r_target)
        res = minimize(portfolio_risk, x0, method='SLSQP', bounds=bounds, constraints=cons)  # type: ignore
        if res.success:
            w_opt = res.x
            frontier_points.append({
                'Return': float(r_target),
                'Risk': float(portfolio_risk(w_opt)),
                'Weights': w_opt
            })
        else:
            logging.error(f"Optimization failed for target return {r_target:.4f}")

    frontier_df = pd.DataFrame(frontier_points)
    return frontier_df

def plot_esg_histogram(esg_df, outputfile, name):
    """
    Plottet ein Histogramm der ESG-Scores mit Binning in 5er-Schritten.

    Input:
        esg_df: DataFrame mit Spalten ['Ticker', 'ESG']
    """

    # ESG-Spalte zu numerisch konvertieren, Fehler -> NaN
    esg_numeric = pd.to_numeric(esg_df['ESG'], errors='coerce')
    esg_numeric = esg_numeric.dropna()  # NaN rauswerfen

    # Bin-Grenzen (0-100 in 5er-Schritten)
    bins = np.arange(0, 105, 5)

    plt.figure(figsize=(10,6))
    plt.hist(esg_numeric, bins=bins, edgecolor='black', alpha=0.7)
    plt.title("Verteilung der ESG-Scores")
    plt.xlabel("ESG Score")
    plt.ylabel("Anzahl der Ticker")
    plt.xticks(bins, rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    #plt.show()
    #Plot speichern
    output_path = outputfile + "/" + name + "/" + "esg_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Speicher freigeben, falls du viele Plots erzeugst


def clean_data(prices, esg):
    corrupt_tickers = []
    #corrupt_tickers = ["COIN.OQ", "NVDA.OQ", "SMCI.OQ", "VST.N"]

    # First mark all tickers without historical data
    for ticker, df in prices.items():
        df['Date'] = pd.to_datetime(df['Date'])
        duplicates = df[df.duplicated(subset=['Date'], keep=False)]
        if not duplicates.empty:
            corrupt_tickers.append(ticker)


    # Second mark all tickers without esg data
    corrupt_esg = esg[pd.to_numeric(esg['ESG'], errors='coerce').isna()]["Ticker"].tolist()
    corrupt_tickers.extend(corrupt_esg)

    # Remove all marked tickers from data
    logging.info("Remove " + str(len(corrupt_tickers)) + " corrupt tickers: " + str(corrupt_tickers))
    for ticker in corrupt_tickers:
        if ticker in prices:
            del prices[ticker]
    esg = esg[~esg['Ticker'].isin(corrupt_tickers)].copy()

    return prices, esg


def adjust_data(prices, esg, border):
    # Filter Ticker im ESG-DataFrame
    esg['ESG'] = pd.to_numeric(esg['ESG'])
    filtered_esg = esg[esg['ESG'] <= border].copy()

    # Ticker-Liste für Filter
    valid_tickers = filtered_esg['Ticker'].tolist()

    # Dict filtern
    filtered_prices = {ticker: df for ticker, df in prices.items() if ticker in valid_tickers}

    return filtered_prices, filtered_esg

def check_and_sort_alignment(prices, esg):
    """
    Prüft, ob die Ticker in prices (Dict) und esg (DataFrame) an den gleichen Positionen übereinstimmen,
    und sortiert beide nach Ticker aufsteigend.

    Returns:
        prices_sorted: Dict nach Ticker sortiert
        esg_sorted: DataFrame nach Ticker sortiert
    """
    # Ticker-Liste für beide Objekte
    prices_tickers = list(prices.keys())
    esg_tickers = esg['Ticker'].tolist()

    # Überprüfung vor Sortierung
    mismatches = []
    for i, (p, e) in enumerate(zip(prices_tickers, esg_tickers)):
        if p != e:
            mismatches.append((i, p, e))
    if mismatches:
        logging.info("Vor Sortierung: Ticker stimmen nicht überein an folgenden Positionen:")
    else:
        logging.info("Alle Ticker stimmen vor Sortierung überein!")

    # Sortieren
    sorted_tickers = sorted(prices_tickers)
    prices_sorted = {ticker: prices[ticker] for ticker in sorted_tickers}
    esg_sorted = esg[esg['Ticker'].isin(sorted_tickers)].copy()
    esg_sorted = esg_sorted.set_index('Ticker').loc[sorted_tickers].reset_index()

    # Überprüfung nach Sortierung
    prices_sorted_list = list(prices_sorted.keys())
    esg_sorted_list = esg_sorted['Ticker'].tolist()
    mismatches_after = [(i, p, e) for i, (p, e) in enumerate(zip(prices_sorted_list, esg_sorted_list)) if p != e]
    if mismatches_after:
        logging.error("Ticker stimmen nach Sortierung immer noch nicht überein:", mismatches_after)
    else:
        logging.info("Alle Ticker stimmen nach Sortierung überein!")

    return prices_sorted, esg_sorted


def calculate_frontiers(prices, esg, freq, n_workers=4):
    """
    Berechnet für verschiedene ESG-Borders die Markowitz-Frontier.

    Inputs:
        prices: Dict mit historischen Preisen
        esg: DataFrame mit ESG-Scores
        n_workers: Anzahl paralleler Threads
    Output:
        frontiers: Dict mit ESG-Border als Key und Frontier DataFrame als Value
    """

    frontiers = {}
    esg_borders = list(range(100, 49, -5))

    def process_border(border):
        logging.info(f"Adjust Data for ESG-Border: {border}")
        border_historical_prices, border_esg_scores = adjust_data(prices, esg, border)
        checked_historical_prices, checked_esg_scores = check_and_sort_alignment(border_historical_prices,
                                                                                 border_esg_scores)
        logging.info(f"Calculate Co-Variance and get annual returns for ESG-Border: {border}")
        cov, returns = calculate_cov_matrix(checked_historical_prices, freq)
        #annual_returns_sorted = returns.sort_values(ascending=False)
        #print(annual_returns_sorted)
        #exit(1)

        logging.info(f"Calculate Markowitz Frontier for ESG-Border: {border}")
        mark_frontier = markowitz_frontier(returns, cov)
        return border, mark_frontier

    # Multithreading mit Fortschrittsanzeige
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_border, border): border for border in esg_borders}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Calculating Frontiers"):
            esg_border, frontier = future.result()
            frontiers[esg_border] = frontier

    return frontiers


def plot_all_frontiers(frontiers_dict, outputfile, name):
    """
    Plottet mehrere Markowitz-Frontiers aus einem Dict.

    Inputs:
        frontiers_dict: Dict, keys = ESG-Borders, values = DataFrames mit ['Return', 'Risk', 'Weights']
    """
    plt.figure(figsize=(12, 8))

    n = len(frontiers_dict)
    cmap = plt.get_cmap("viridis")  # Colormap holen
    colors = cmap(np.linspace(0, 1, n))

    for color, (esg_border, frontier_df) in zip(colors, sorted(frontiers_dict.items())):
        plt.plot(frontier_df['Risk'], frontier_df['Return'], label=f'ESG ≤ {esg_border}', color=color)

    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Expected Return")
    plt.title("Markowitz Efficient Frontiers für verschiedene ESG-Borders")
    plt.legend()
    plt.grid(True)

    # plt.show()
    # Plot speichern
    output_path = outputfile + "/" + name + "/" + "frontiers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Speicher freigeben, falls du viele Plots erzeugst

def save_dfs_to_excel(outputfile, name, tickers, historical_prices, esg_scores, clean_historical_prices, clean_esg_scores, frontiers, cov, returns):
    # Pfad zur Datei erstellen
    path = outputfile + "/" + name + "/" + "Data.xlsx"

    # Listen vorbereiten (Dict to DF)
    dfs_historical_prices = []
    for ticker, df in historical_prices.items():
        df_copy = df.copy()
        df_copy['Ticker'] = ticker
        dfs_historical_prices.append(df_copy)
    combined_df_historical_prices = pd.concat(dfs_historical_prices, ignore_index=True)
    dfs_clean_historical_prices = []
    for ticker, df in clean_historical_prices.items():
        df_copy = df.copy()
        df_copy['Ticker'] = ticker
        dfs_clean_historical_prices.append(df_copy)
    combined_df_clean_historical_prices = pd.concat(dfs_clean_historical_prices, ignore_index=True)
    dfs_frontiers = []
    for border, df in frontiers.items():
        df_copy = df.copy()
        df_copy['ESG-Border'] = border
        dfs_frontiers.append(df_copy)
    combined_df_frontiers = pd.concat(dfs_frontiers, ignore_index=True)

    # ExcelWriter verwenden, um mehrere Sheets zu schreiben
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        tickers.to_excel(writer, sheet_name='Tickers', index=True)
        returns.to_excel(writer, sheet_name='Annual-Returns', index=True)
        combined_df_historical_prices.to_excel(writer, sheet_name='Historical-Prices', index=True)
        combined_df_clean_historical_prices.to_excel(writer, sheet_name='Clean-Historical-Prices', index=True)
        esg_scores.to_excel(writer, sheet_name='ESG-Scores', index=True)
        clean_esg_scores.to_excel(writer, sheet_name='Clean-ESG-Scores', index=True)
        cov.to_excel(writer, sheet_name='Co-Variance', index=True)
        combined_df_frontiers.to_excel(writer, sheet_name='Frontiers', index=True)


def main(load_from_file, universe, start, end, freq, outputfile, name):
    file_historical_prices = outputfile + '/' + name + '/' + 'historical_prices.pkl'
    file_esg_scores = outputfile + '/' + name + '/' + 'esg_scores.pkl'

    logging.info("Open Refinitiv Session")
    open_ld()

    logging.info("Get Tickers")
    tickers = get_tickers(universe, start)

    if load_from_file and os.path.exists(file_historical_prices):
        logging.info(f"Loading historical prices from {file_historical_prices}")
        with open(file_historical_prices, 'rb') as f:
            historical_prices = pickle.load(f)
    else:
        logging.info("Get historical prices")
        historical_prices = get_historical_price(tickers, start, end)
        # Pickle speichern
        with open(file_historical_prices, 'wb') as f:
            pickle.dump(historical_prices, f)   # type: ignore
        logging.info(f"Saved historical prices to {file_historical_prices}")

    if load_from_file and os.path.exists(file_esg_scores):
        logging.info(f"Loading ESG scores from {file_esg_scores}")
        with open(file_esg_scores, 'rb') as f:
            esg_scores = pickle.load(f)
    else:
        logging.info("Get ESG scores")
        esg_scores = get_esg_scores(tickers, end)
        with open(file_esg_scores, 'wb') as f:
            pickle.dump(esg_scores, f)          # type: ignore
        logging.info(f"Saved ESG scores to {file_esg_scores}")

    logging.info("Clean up Data")
    clean_historical_prices, clean_esg_scores = clean_data(historical_prices, esg_scores)

    # Export Prices to debug
    excelname = "prices-" + str(random.randint(1, 9999)) + ".xlsx"
    dfs_historical_prices = []
    for ticker, df in clean_historical_prices.items():
        df_copy = df.copy()
        df_copy['Ticker'] = ticker
        dfs_historical_prices.append(df_copy)
    combined_df_historical_prices = pd.concat(dfs_historical_prices, ignore_index=True)
    with pd.ExcelWriter(excelname, engine='xlsxwriter') as writer:
        combined_df_historical_prices.to_excel(writer, sheet_name='Prices', index=True)

    logging.info("Get Efficient Frontier for several ESG-Borders")
    cov, returns = calculate_cov_matrix(clean_historical_prices, freq)
    frontiers = calculate_frontiers(clean_historical_prices, clean_esg_scores, freq)
    plot_esg_histogram(clean_esg_scores, outputfile, name)
    plot_all_frontiers(frontiers, outputfile, name)

    logging.info("Save Data to Excel")
    save_dfs_to_excel(outputfile, name, tickers, historical_prices, esg_scores, clean_historical_prices, clean_esg_scores, frontiers, cov, returns)

    ld.close_session()

if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)

    # YAML-Konfiguration laden
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    runs = config.get("runs", [])
    if not runs:
        logging.error("Keine Runs in config.yaml gefunden!")
        exit(1)

    # Kommandozeilenparameter parsen
    parser = argparse.ArgumentParser(description="Run Efficient Frontier Pipeline")
    parser.add_argument(
        '--load_from_file',
        action='store_true',
        help='Load historical prices from pickle instead of fetching from API'
    )
    args = parser.parse_args()

    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        filename= "runtime.log",
        filemode="w",
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # Run Ordner anlegen
    rootfolder = datetime.now().strftime("%Y-%m-%d_%H-%M") + "_Output/"
    os.makedirs(rootfolder, exist_ok=True)

    # Alle Runs nacheinander ausführen
    for run in runs:
        # Ordner anlegen
        folder = rootfolder + run['name']
        os.makedirs(folder, exist_ok=True)

        logging.info(f"===== Starte Run: {run['name']} =====")
        main(
                load_from_file=args.load_from_file,
                universe=run["universe"],
                start=run["start"],
                end=run["end"],
                freq=run["freq"],
                outputfile=rootfolder,
                name=run["name"]
        )
        logging.info(f"===== Run {run['name']} beendet =====\n")