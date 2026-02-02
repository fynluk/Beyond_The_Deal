import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ======================
# Asset-Daten
# ======================
returns = np.array([0.20, 0.15])
risks = np.array([0.10, 0.05])
risk_free_rate = 0.025

# Kovarianzmatrix (Korrelation = 0)
cov_matrix = np.array([
    [risks[0] ** 2, 0],
    [0, risks[1] ** 2]
])

# Gewichte
weights = np.linspace(0, 1, 100)

portfolio_returns = []
portfolio_risks = []
sharpe_ratios = []

for w in weights:
    w_vec = np.array([w, 1 - w])

    port_return = np.dot(w_vec, returns)
    port_risk = np.sqrt(w_vec.T @ cov_matrix @ w_vec)
    sharpe = (port_return - risk_free_rate) / port_risk

    portfolio_returns.append(port_return)
    portfolio_risks.append(port_risk)
    sharpe_ratios.append(sharpe)

portfolio_returns = np.array(portfolio_returns)
portfolio_risks = np.array(portfolio_risks)
sharpe_ratios = np.array(sharpe_ratios)

# ======================
# Grafik 1: Efficient Frontier
# ======================
plt.figure(figsize=(7, 5))
plt.plot(portfolio_risks, portfolio_returns, linewidth=2)

plt.xlabel("Risk")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")

plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.grid(True)
plt.show()

# ======================
# Grafik 2: Capital Market Line (Sharpe Ratio)
# ======================
max_sharpe_idx = np.argmax(sharpe_ratios)
max_sharpe = sharpe_ratios[max_sharpe_idx]

# CML
sigma_cml = np.linspace(0, max(portfolio_risks), 100)
return_cml = risk_free_rate + max_sharpe * sigma_cml

plt.figure(figsize=(7, 5))
plt.plot(portfolio_risks, portfolio_returns, linewidth=2, label="Efficient Frontier")
plt.plot(sigma_cml, return_cml, linestyle="--", linewidth=2, label="Capital Market Line")

# Call-out Box oben rechts
plt.annotate(
    f"Max Sharpe Ratio:\n{max_sharpe:.2f}",
    xy=(0.98, 0.95),
    xytext=(0.98, 0.95),
    textcoords="axes fraction",
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", fc="white")
)

plt.xlabel("Risk")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier & Capital Market Line")

plt.gca().xaxis.set_major_formatter(PercentFormatter(1))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.legend()
plt.grid(True)
plt.show()