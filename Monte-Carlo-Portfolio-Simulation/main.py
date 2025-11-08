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


def refinitiv_session():
    config = ld.get_config()
    config.set_param("logs.transports.console.enabled", True)
    config.set_param("logs.level", "info")
    ld.open_session()

def get_tickers():
    universe = '0#.SPX'     # S&P 500
    #universe = '0#.FTMC'    # FTSE 250
    #universe = '0#.FTLC'    # FTSE 350
    #universe = '0#.STOXX'   # STOXX Europe 600
    #universe = '0#.RUT'     # Russell 2000
    #universe = '0#.RUA'     # Russell 3000
    query = ld.get_data(
        universe=universe,
        fields=['TR.RIC'],
        parameters={
            'SDate': '2025-06-30'
        }
    )
    del query['RIC']

    return query['Instrument'].dropna().to_list()

def get_data(tickers):
    prices = ld.get_data(
        universe=tickers,
        fields=['TR.PriceClose', 'TR.PriceClose.date'],
        parameters={
            'Frq': 'D',
            'SDate': '2023-07-01',
            'EDate': '2025-06-30'
        }
    )

    esg = ld.get_data(
        universe=tickers,
        fields=['TR.TRESGScore'],
        parameters={
            'SDate': '2023-07-01'
        }
    )

    return prices, esg

def review_data(prices, esg):
    missing_prices = prices[prices[['Price Close', 'Date']].isna().any(axis=1)]
    tickers_with_missing_data = list(missing_prices['Instrument'].unique())

    missing_esg = esg[esg['ESG Score'].isna()]['Instrument'].tolist()
    tickers_with_missing_data.extend(missing_esg)

    return list(set(tickers_with_missing_data))

def clean_data(prices, esg, tickers_to_remove):
    prices_clean = prices[~prices['Instrument'].isin(tickers_to_remove)].copy()
    prices_clean = prices_clean.drop_duplicates(subset=['Date', 'Instrument'])

    esg_clean = esg[~esg['Instrument'].isin(tickers_to_remove)].copy()

    return prices_clean, esg_clean


def plot_equity_return_esg(clean_prices, clean_esg_score):
    # Pivot: jede Aktie als Spalte, Index = Datum
    price_pivot = clean_prices.pivot(index='Date', columns='Instrument', values='Price Close')
    price_pivot.index = pd.to_datetime(price_pivot.index)

    # Prozentuale Änderung und kumulativer Return
    returns = price_pivot.pct_change().dropna()
    cum_returns = (1 + returns).cumprod() - 1
    final_returns = cum_returns.iloc[-1]

    # ESG Scores als Series
    if isinstance(clean_esg_score, pd.DataFrame):
        if 'Instrument' in clean_esg_score.columns and 'ESG Score' in clean_esg_score.columns:
            esg_series = clean_esg_score.set_index('Instrument')['ESG Score']
        else:
            raise ValueError("clean_esg_score DataFrame muss 'Instrument' und 'ESG Score' Spalten haben")
    else:
        esg_series = clean_esg_score

    # Daten kombinieren
    data = pd.DataFrame({
        'ESG Score': esg_series,
        'Return': final_returns
    }).dropna()

    # Lineare Regression
    slope, intercept, r_value, p_value, std_err = linregress(data['ESG Score'], data['Return'])
    line_x = np.linspace(data['ESG Score'].min(), data['ESG Score'].max(), 100)
    line_y = intercept + slope * line_x

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(data['ESG Score'], data['Return'], color='blue', label='Aktien')
    plt.plot(line_x, line_y, color='red', label=f'Trendlinie\nSlope={slope:.4f}, R²={r_value**2:.4f}')
    plt.xlabel('ESG Score')
    plt.ylabel('Return')
    plt.title('Return vs. ESG Score')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_returns(clean_prices, freq):
    price_pivot = clean_prices.pivot(index='Date', columns='Instrument', values='Price Close')
    price_pivot_resampled = price_pivot.resample(freq).last()

    returns = price_pivot_resampled.pct_change()

    return price_pivot, price_pivot_resampled, returns


def _simulate_portfolio(idx, returns, esg_dict, instruments, rand_seed):
    np.random.seed(rand_seed + idx)
    weights = np.random.random(len(instruments))
    weights /= np.sum(weights)

    port_rets = (returns * weights).sum(axis=1)
    cumulative_return = (1 + port_rets).prod() - 1
    vol = port_rets.std()
    port_esg = np.sum([weights[i] * esg_dict[instr] for i, instr in enumerate(instruments)])
    weights_dict = {instr: weights[i] for i, instr in enumerate(instruments)}

    return {'Portfolio_Return': cumulative_return,
            'Volatility': vol,
            'ESG_Score': port_esg,
            **weights_dict}


def monte_carlo_portfolios(returns, esg, num_portfolios, rand_seed, max_workers):
    instruments = returns.columns.tolist()
    esg_dict = esg.set_index('Instrument')['ESG Score'].to_dict()

    # functools.partial bindet die festen Argumente
    simulate_partial = partial(_simulate_portfolio, returns=returns, esg_dict=esg_dict,
                               instruments=instruments, rand_seed=rand_seed)

    portfolio_results = process_map(
        simulate_partial,
        range(num_portfolios),
        max_workers=max_workers,
        desc="Simulating Monte Carlo Portfolios",
        unit="portfolio",
        chunksize=10
    )

    return pd.DataFrame(portfolio_results)

def plot_portfolio_return_esg(portfolio_results):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        x=portfolio_results['ESG_Score'],
        y=portfolio_results['Portfolio_Return'],
        alpha=0.7
    )
    plt.xlabel('ESG Score')
    plt.ylabel('Portfolio Return')
    plt.title(f'Portfolio Return vs. ESG Score')
    plt.grid(True)
    plt.show()

def plot_portfolio_heatmap(portfolio_results):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        x=portfolio_results['Volatility'],
        y=portfolio_results['Portfolio_Return'],
        c=portfolio_results['ESG_Score'],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='ESG_Score')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.title(f'Monte-Carlo Portfolio Heatmap')
    plt.grid(True)
    plt.show()

def plot_portfolio_3d(portfolio_results):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        portfolio_results['ESG_Score'],
        portfolio_results['Volatility'],
        portfolio_results['Portfolio_Return'],
        c=portfolio_results['ESG_Score'],
        cmap='viridis',
        alpha=0.8
    )

    ax.set_ylabel('Volatility')
    ax.set_zlabel('Return')
    ax.set_xlabel('ESG Score')
    ax.set_title('Monte-Carlo Portfolio 3D Visualization')

    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('ESG_Score')

    plt.show()


def plot_portfolio_surface(portfolio_results, method='linear', grid_size=500):
    y = portfolio_results['Volatility'].values
    z = portfolio_results['Portfolio_Return'].values
    x = portfolio_results['ESG_Score'].values

    # Grid für Oberfläche erstellen
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolation auf Grid
    ZI = griddata((x, y), z, (XI, YI), method=method)

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(XI, YI, ZI, cmap='viridis', alpha=0.8, edgecolor='none')

    ax.set_ylabel('Volatility')
    ax.set_zlabel('Return')
    ax.set_xlabel('ESG Score')
    ax.set_title('Monte-Carlo Portfolio Surface')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Return')
    plt.show()

def plot_portfolio_surface_interactive(portfolio_results, method='linear', grid_size=100):
    # Originaldaten
    x = portfolio_results['ESG_Score'].values
    y = portfolio_results['Volatility'].values
    z = portfolio_results['Portfolio_Return'].values

    # Grid für Oberfläche erstellen
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolation auf Grid
    ZI = griddata((x, y), z, (XI, YI), method=method)

    # Interaktiver 3D-Plot
    fig = go.Figure(data=[go.Surface(
        x=XI,
        y=YI,
        z=ZI,
        colorscale='Viridis',
        colorbar=dict(title='Return')
    )])

    fig.update_layout(
        title='Monte-Carlo Portfolio Surface',
        scene=dict(
            xaxis_title='ESG Score',
            yaxis_title='Volatility',
            zaxis_title='Portfolio Return'
        ),
        autosize=True
    )

    fig.show()

def main():
    refinitiv_session()
    tickers = get_tickers()
    prices, esg = get_data(tickers)
    tickers_to_remove = review_data(prices, esg)
    clean_prices, clean_esg = clean_data(prices, esg, tickers_to_remove)
    plot_equity_return_esg(clean_prices, clean_esg)
    pivot1, pivot2, returns = calculate_returns(clean_prices, freq='W')
    portfolio_results = monte_carlo_portfolios(returns, esg, num_portfolios=100000, rand_seed=1337, max_workers=8)
    plot_portfolio_return_esg(portfolio_results)
    plot_portfolio_heatmap(portfolio_results)
    plot_portfolio_3d(portfolio_results)
    plot_portfolio_surface(portfolio_results)
    plot_portfolio_surface_interactive(portfolio_results)

    dataframes_to_save = [
        ('01-Prices', prices),
        ('02-ESG_Scores', esg),
        ('03-Clean_Prices', clean_prices),
        ('04-Clean_ESG_Scores', clean_esg),
        ('05-Pivot_Daily', pivot1),
        ('06-Pivot_Weekly', pivot2),
        ('07-Daily_Returns', returns),
        ('08-Monte_Calo_Portfolios', portfolio_results)
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