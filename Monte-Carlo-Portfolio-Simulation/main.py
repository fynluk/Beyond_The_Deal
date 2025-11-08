import logging
import lseg.data as ld
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


def refinitiv_session():
    ld.open_session()

def get_tickers():
    query = ld.get_data(
        universe='0#.SPX',
        fields=['TR.IndexConstituentRIC'],
        parameters={
            'SDate': '2025-10-31'
        }
    )
    return query['Instrument'].dropna().to_list()[:5]
    #return query['Instrument'].dropna().to_list()

def get_data(tickers):
    prices = ld.get_data(
        universe=tickers,
        fields=['TR.PriceClose', 'TR.PriceClose.date'],
        parameters={
            'Frq': 'D',
            'SDate': '2023-11-01',
            'EDate': '2025-10-31'
        }
    )

    esg = ld.get_data(
        universe=tickers,
        fields=['TR.TRESGScore'],
        parameters={
            'SDate': '2023-11-01'
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
    esg_clean = esg[~esg['Instrument'].isin(tickers_to_remove)].copy()

    return prices_clean, esg_clean


def calculate_returns(clean_prices, freq):
    price_pivot = clean_prices.pivot(index='Date', columns='Instrument', values='Price Close')
    display(price_pivot)

    price_pivot_resampled = price_pivot.resample(freq).last()
    display(price_pivot_resampled)

    returns = price_pivot_resampled.pct_change()

    return price_pivot, price_pivot_resampled, returns


def monte_carlo_portfolios(returns, esg, num_portfolios, rand_seed):
    np.random.seed(rand_seed)

    instruments = returns.columns.tolist()
    esg_dict = esg.set_index('Instrument')['ESG Score'].to_dict()

    portfolio_results = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(instruments))
        weights /= np.sum(weights)

        port_rets = (returns * weights).sum(axis=1)

        avg_return = port_rets.mean()
        vol = port_rets.std()
        port_esg = np.sum([weights[i] * esg_dict[instr] for i, instr in enumerate(instruments)])

        portfolio_results.append({
            'Avg_Return': avg_return,
            'Volatility': vol,
            'ESG_Score': port_esg,
            'Weights': weights
        })

    return pd.DataFrame(portfolio_results)


def plot_portfolio_heatmap(portfolio_results):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        x=portfolio_results['Volatility'],
        y=portfolio_results['ESG_Score'],
        c=portfolio_results['Avg_Return'],
        cmap='viridis',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Avg_Return')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio ESG Score')
    plt.title(f'Monte-Carlo Portfolio Avg_Return Heatmap')
    plt.grid(True)
    plt.show()

def plot_portfolio_3d(portfolio_results):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        portfolio_results['Volatility'],
        portfolio_results['Avg_Return'],
        portfolio_results['ESG_Score'],
        c=portfolio_results['ESG_Score'],
        cmap='viridis',
        alpha=0.8
    )

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_zlabel('ESG Score')
    ax.set_title('Monte-Carlo Portfolio 3D Visualization')

    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('ESG_Score')

    plt.show()


def plot_portfolio_surface(portfolio_results, method='linear', grid_size=500):
    # x = Volatility, y = Return, z = ESG Score
    x = portfolio_results['Volatility'].values
    y = portfolio_results['Avg_Return'].values
    z = portfolio_results['ESG_Score'].values

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

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.set_zlabel('ESG Score')
    ax.set_title('Monte-Carlo Portfolio Surface')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='ESG Score')
    plt.show()


def main():
    refinitiv_session()
    tickers = get_tickers()
    prices, esg = get_data(tickers)
    tickers_to_remove = review_data(prices, esg)
    clean_prices, clean_esg = clean_data(prices, esg, tickers_to_remove)
    pivot1, pivot2, returns = calculate_returns(clean_prices, freq='W')
    portfolio_results = monte_carlo_portfolios(returns, esg, num_portfolios=5000, rand_seed=1337)
    plot_portfolio_heatmap(portfolio_results)
    plot_portfolio_3d(portfolio_results)
    plot_portfolio_surface(portfolio_results)

    with pd.ExcelWriter("Data.xlsx", engine='xlsxwriter') as writer:
        prices.to_excel(writer, sheet_name='Prices', index=True)
        esg.to_excel(writer, sheet_name='ESG', index=True)
        clean_prices.to_excel(writer, sheet_name='Clean Prices', index=True)
        clean_esg.to_excel(writer, sheet_name='Clean ESG', index=True)
        pivot1.to_excel(writer, sheet_name='Pivot 1', index=True)
        pivot2.to_excel(writer, sheet_name='Pivot 2', index=True)
        returns.to_excel(writer, sheet_name='Returns', index=True)

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