#%%
import numpy as np
import pandas as pd
from scipy.stats import norm


def monte_carlo_merton(V_0, K, r, sigma, T, M):
    
    W_T = np.random.randn(M)*np.sqrt(T)
    V_T = V_0 * np.exp((r - 0.5 * sigma**2) * T + sigma * W_T)

    S_T = np.maximum(V_T - K,0)
    B_T = V_T - S_T

    Equity_sim_mean = np.mean(S_T)
    Debt_sim_mean = np.mean(B_T)

    defaulted_simulations = np.sum(V_T < K)
    prob_default = (defaulted_simulations/M)*100
    

    return Equity_sim_mean, Debt_sim_mean, prob_default
