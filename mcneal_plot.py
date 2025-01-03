#%%
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import numpy as np
from analytical_functions import credit_spread_model

#V= 3829968375.7
#K = 931124000.0
V=100
K=60
r = 0
T= 2
t = 0
sigma = 0.25

def plot_spread_vol(V,K,r,T,t):
    vol_interval = np.linspace(0.01,0.5,1000)
    credit_spread_vol = [credit_spread_model(V,K,sigma,r,T,t) for sigma in vol_interval]

    plt.figure(figsize=(12,6))
    plt.plot(vol_interval,credit_spread_vol,label = 'Credit Spread')
    plt.xlabel('Volatility (sigma)')
    plt.ylabel('Credit Spread (%)')
    plt.title('Impact of Volatility on Credit Spread')
    plt.grid(False)
    plt.show()

def plot_spread_time(V,K,r,T,t):
    time_interval = np.linspace(0,5,1000)
    credit_spread_time = [credit_spread_model(V,K,sigma,r,T,t) for T in time_interval]

    plt.figure(figsize=(12,6))
    plt.plot(time_interval,credit_spread_time,label ='Credit Spread')
    plt.xlabel("Time to Maturity")
    plt.ylabel("Credit Spread (%)")
    plt.title("Impact of Maturity on Credit Spread")
    plt.grid(False)
    plt.show()