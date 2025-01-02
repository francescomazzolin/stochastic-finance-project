#%%
import numpy as np
import matplotlib.pyplot as plt

def plot_asset_paths_with_default(threshold, asset_paths, time_horizon, num_paths, instrument):
    
    time_steps = asset_paths.shape[1]
    time = np.linspace(0, time_horizon, time_steps)

    plt.figure(figsize=(12, 8))
    for i in range(num_paths):
        plt.plot(time, asset_paths[i, :], alpha=0.7, linewidth=1.0)

    # Plot the debt threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Debt Threshold (D)')

    # Highlight the default area
    plt.fill_between(time, 0, threshold, color='red', alpha=0.2, label='Default Region')

    # Customize the plot
    plt.title(f"Simulated Asset Paths and Default Threshold for {instrument}", fontsize=14)
    plt.xlabel("Time (Years)", fontsize=12)
    plt.ylabel("Asset Value", fontsize=12)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example Usage
def monte_carlo_simulation_paths(V0, sigma, r, T, M, N):
    """
    Generate Monte Carlo simulated asset paths using geometric Brownian motion.

    Parameters:
    - V0 (float): Initial asset value.
    - sigma (float): Asset volatility.
    - r (float): Risk-free rate.
    - T (float): Time horizon (years).
    - M (int): Number of paths.
    - N (int): Number of time steps.

    Returns:
    - np.ndarray: Simulated asset paths (shape: [M, N]).
    """
    dt = T / N
    paths = np.zeros((M, N))
    paths[:, 0] = V0
    for t in range(1, N):
        z = np.random.standard_normal(M)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return paths

# Parameters
V0 = 164475000000  # Initial asset value (Apple)
D = 107525000000   # Debt threshold
sigma = 0.177826   # Volatility
r = 0.04           # Risk-free rate
T = 1              # Time horizon (1 year)
M = 1000           # Number of Monte Carlo paths
N = 252            # Number of time steps (daily)

# Generate simulated paths
asset_paths = monte_carlo_simulation_paths(V0, sigma, r, T, M, N)

# Plot the asset paths and default threshold
plot_asset_paths_with_default(D, asset_paths, T, num_paths=1000, instrument="AAPL.O")
