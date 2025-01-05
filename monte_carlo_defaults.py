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


def merton_jumps_default(V0, K, T, M, N, lam, m, v, r, sigma):
   
    dt = T / N  
    paths = np.zeros((M, N))  
    paths[:, 0] = V0  

    for t in range(1, N):
        z = np.random.standard_normal(M)  
        jump_counts = np.random.poisson(lam * dt, M)  
        jump_sizes = np.random.normal(m, v, M) 
        
        
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        ) * np.exp(jump_counts * jump_sizes)

   
    V_T = paths[:, -1]  
    S_T = np.maximum(V_T - K, 0)  
    B_T = V_T - S_T  

    
    defaulted_paths = np.any(paths < K, axis=1)
    print(f"Number of default paths: {np.sum(defaulted_paths)}")
    prob_default = (np.sum(defaulted_paths) / M) * 100  

    Equity_mean = np.mean(S_T)
    Debt_mean = np.mean(B_T)

    return Equity_mean, Debt_mean, prob_default

