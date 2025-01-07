import matplotlib.pyplot as plt
import numpy as np
from analytical_functions import credit_spread_model

"""
This function plots the impact of VOLATILITY on the credit spread according to 
the Merton's model
"""

def plot_spread_vol(V, K, r, T, t, debug=False):
    vol_interval = np.linspace(0.01, 0.5, 1000)
    credit_spread_vol = [credit_spread_model(V, K, sigma, r, T, t) for sigma in vol_interval]
    credit_spread_vol_percentage = [cs * 100 for cs in credit_spread_vol]

    plt.figure(figsize=(12, 6))
    plt.plot(vol_interval, credit_spread_vol_percentage, label='Credit Spread')
    plt.xlabel('Volatility (σ)')  
    plt.ylabel('Credit Spread (%)')  
    plt.title('Theoretical Impact of Volatility on Credit Spread')
    plt.grid(False)
    plt.legend()
    plt.show()


"""
This function plots the impact of MATURITY on the credit spread according to 
the Merton's model
"""

def plot_spread_time(V, K, r, T, t, sigma=0.25, debug=False):
    time_interval = np.linspace(0.01, 5, 1000)  
    credit_spread_time = [credit_spread_model(V, K, sigma, r, T, t) for T in time_interval]
    credit_spread_time_percentage = [cs * 100 for cs in credit_spread_time]

    plt.figure(figsize=(12, 6))
    plt.plot(time_interval, credit_spread_time_percentage, label='Credit Spread')
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Credit Spread (%)") 
    plt.title("Theoretical Impact of Maturity on Credit Spread")
    plt.grid(False)
    plt.legend()
    plt.show()

"""
This function plots the various stochastic processes simulated paths, for a given asset 
and a given maturity against the threshold below which the Merton's model postulates a default
"""

def plot_asset_paths_with_default(threshold, asset_paths, time_horizon, instrument, log = False):
    
    print(f"Min asset: {np.min(asset_paths[:, -1])}")
    print(f"Max asset: {np.max(asset_paths[:, -1])}")
    
    if log:
        
        threshold = np.log(threshold)

    print(f'With threshold: {threshold}')

    #Design variables

    final_values = asset_paths[:,-1]

    rng = max(final_values) - min(final_values)

    time_steps = asset_paths.shape[1]
    time = np.linspace(0, time_horizon, time_steps)

    num_paths = asset_paths.shape[0]

    plt.figure(figsize=(12, 8))
    for i in range(num_paths):
        plt.plot(time, asset_paths[i, :], alpha=0.7, linewidth=1.0)

    # Plot the debt threshold line
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Debt Threshold (D)')

    # Highlight the default area
    plt.fill_between(time, threshold - rng, threshold, color='red', alpha=0.2, label='Default Region')

    # Customize the plot
    plt.title(f"Simulated Asset Paths and Default Threshold for {instrument}", fontsize=14)
    plt.xlabel("Time (Years)", fontsize=12)
    plt.ylabel("Asset Value", fontsize=12)
    #plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def merton_jumps_plot(V0, sigma, r, T, M, N, lam, m, v, K, instrument):
   
    dt = T / N  
    paths = np.zeros((M, N))  
    paths[:, 0] = V0  
    time = np.linspace(0, T, N)
    for t in range(1, N):
        
        z = np.random.standard_normal(M)
        jump_count = np.random.poisson(lam * dt, M)
        jump_sizes = np.sum(np.random.normal(m, v, (M, max(jump_count))) * (np.arange(max(jump_count)) < jump_count[:, None]), axis=1)
        
        # Geometric Brownian motion + jumps
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z + jump_sizes)

    # Plotting the paths
    plt.figure(figsize=(10, 6))
    for i in range(M):
        plt.plot(np.linspace(0, T, N), paths[i], lw=0.8)
    plt.axhline(y=K, color='r', linestyle='--', linewidth=2, label='Debt Threshold (D)')
    plt.fill_between(time,0, K, color='red', alpha=0.2, label='Default Region')
    plt.title(f"Merton Jump-Diffusion Model Simulated Paths for {instrument}", fontsize = 14)
    plt.xlabel("Time (Years)")
    plt.ylabel("Asset Value")
    plt.show()

    return None                                       
