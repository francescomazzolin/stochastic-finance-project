import matplotlib.pyplot as plt
import numpy as np
from analytical_functions import credit_spread_model

def plot_spread_vol(V, K, r, T, t, debug=False):
    vol_interval = np.linspace(0.01, 0.5, 1000)
    credit_spread_vol = [credit_spread_model(V, K, sigma, r, T, t) for sigma in vol_interval]
    credit_spread_vol_percentage = [cs * 100 for cs in credit_spread_vol]

    plt.figure(figsize=(12, 6))
    plt.plot(vol_interval, credit_spread_vol_percentage, label='Credit Spread')
    plt.xlabel('Volatility (σ)')
    plt.ylabel('Credit Spread (%)')  
    plt.title('Impact of Volatility on Credit Spread')
    plt.grid(False)
    plt.legend()
    plt.show()

def plot_spread_time(V, K, r, T, t, sigma=0.25, debug=False):
    time_interval = np.linspace(0.01, 5, 1000)  
    credit_spread_time = [credit_spread_model(V, K, sigma, r, T, t) for T in time_interval]
    credit_spread_time_percentage = [cs * 100 for cs in credit_spread_time]

    plt.figure(figsize=(12, 6))
    plt.plot(time_interval, credit_spread_time_percentage, label='Credit Spread')
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Credit Spread (%)") 
    plt.title("Impact of Maturity on Credit Spread")
    plt.grid(False)
    plt.legend()
    plt.show()