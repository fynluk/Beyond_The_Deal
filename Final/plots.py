import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ======================
# Asset-Daten
# ======================
returns = np.array([0.20, 0.08])
risks = np.array([0.40, 0.20])
risk_free_rate = 0.025

# Farben (RGB normalisiert)
frontier_color = (0 / 255, 39 / 255, 80 / 255)
sharpe_color = (245 / 255, 158 / 255, 0 / 255)
grid_color = (236 / 255, 237 / 255, 239 / 255)

# Kovarianzmatrix (Korrelation = 0)
cov_matrix = np.array([
    [risks[0] ** 2, 0],
    [0, risks[1] ** 2]
])

# Gewichte
weights = np.linspace(0, 1, 200)

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

# Extrempunkte
risk_A, return_A = risks[0], returns[0]
risk_B, return_B = risks[1], returns[1]

# ======================
# Ordner für Plots
# ======================
os.makedirs("Plots", exist_ok=True)

# ======================
# Grafik 1: Efficient Frontier
# ======================
plt.figure(figsize=(7, 5))
plt.plot(
    portfolio_risks,
    portfolio_returns,
    color=frontier_color,
    linewidth=2
)

# Extrempunkte plotten
plt.scatter(
    [risk_A, risk_B],
    [return_A, return_B],
    color=frontier_color,
    zorder=5
)

# Beschriftungen
plt.text(risk_A + 0.01, return_A - 0.015, "(100% A, 0% B)", color=frontier_color)
plt.text(risk_B + 0.01, return_B + 0.005, "(0% A, 100% B)", color=frontier_color)

plt.xlabel("Risk")
plt.ylabel("Expected Return")

plt.xlim(0, 0.60)
plt.ylim(0, 0.30)

plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

plt.grid(True, color=grid_color)
plt.savefig("Plots/00-1-efficient_frontier.png", dpi=500, bbox_inches="tight")
plt.show()

# ======================
# Grafik 2: Efficient Frontier + Capital Market Line
# ======================
max_sharpe_idx = np.argmax(sharpe_ratios)
max_sharpe = sharpe_ratios[max_sharpe_idx]

# Capital Market Line
sigma_cml = np.linspace(0, 0.60, 200)
return_cml = risk_free_rate + max_sharpe * sigma_cml

plt.figure(figsize=(7, 5))
plt.plot(
    portfolio_risks,
    portfolio_returns,
    color=frontier_color,
    linewidth=2,
    label="Efficient Frontier"
)

plt.plot(
    sigma_cml,
    return_cml,
    color=sharpe_color,
    linewidth=2,
    linestyle="--",
    label="Capital Market Line"
)

# Extrempunkte erneut plotten
plt.scatter(
    [risk_A, risk_B],
    [return_A, return_B],
    color=frontier_color,
    zorder=5
)

plt.text(risk_A + 0.01, return_A - 0.015, "(100% A, 0% B)", color=frontier_color)
plt.text(risk_B + 0.01, return_B + 0.005, "(0% A, 100% B)", color=frontier_color)

# Sharpe Ratio Call-out
plt.annotate(
    f"SR = {max_sharpe:.2f}",
    xy=(0.98, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
    bbox=dict(boxstyle="round", fc="white")
)

plt.xlabel("Risk")
plt.ylabel("Expected Return")

plt.xlim(0, 0.60)
plt.ylim(0, 0.30)

plt.gca().xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

plt.grid(True, color=grid_color)
plt.legend()
plt.savefig("Plots/00-2-efficient_frontier_cml.png", dpi=500, bbox_inches="tight")
plt.show()
