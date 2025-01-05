import numpy as np 


def monte_carlo_merton_2(V_0, K, r, sigma, T, M):
    #np.random.seed(42)
    print(f'Sigma is: {sigma}')
    W_T = np.random.randn(M) * np.sqrt(T) * sigma
    print(f'The realized sigma is: ({(np.std(W_T))}')
    print(len(W_T))
    log_V_T = np.log(V_0) + ((-0.5 * sigma**2) * T + sigma * W_T)
    #V_T = V_0 * np.exp((-0.5 * sigma**2) * T + sigma * W_T)
    defaulted_simulations = np.sum(log_V_T < np.log(K))
    print(defaulted_simulations)
    prob_default = round((defaulted_simulations / M) * 100, 10)

    return prob_default


def monte_carlo_merton(V_0, K, r, sigma, T, M):
    W_T = np.random.randn(M) * np.sqrt(T)
    log_V_T = np.log(V_0) + ((-0.5 * sigma**2) * T + sigma * W_T)

    #log_V_T_anti = np.log(V_0) + ((-0.5 * sigma**2) * T - sigma * W_T)

    #total_sims = np.concatenate((log_V_T, log_V_T_anti))

    defaulted_simulations = np.sum(log_V_T < np.log(K))

    
    # Debug info
    print(f"Min log(V_T): {np.min(log_V_T)}, Max log(V_T): {np.max(log_V_T)}, Threshold: {np.log(K)}")
    prob_default = (defaulted_simulations / M)
    
    return prob_default



def monte_carlo_path_batched(V_0, K, r, sigma, T, M, steps, batch_size):
    """
    Monte Carlo simulation for Merton model with batched processing.
    
    Parameters:
        V_0: float - Initial firm value.
        K: float - Debt face value (strike).
        r: float - Risk-free rate (unused in real-world drift case).
        sigma: float - Annualized volatility of firm value.
        T: float - Time to maturity in years.
        M: int - Total number of simulations.
        steps: int - Number of steps per path.
        batch_size: int - Number of simulations per batch.
        
    Returns:
        prob_default: float - Default probability (%).
    """
    
    dt = T / steps  # Time increment
    total_defaulted = 0  # Counter for total defaults
    
    # Process in batches
    for _ in range(0, M, batch_size):
        current_batch_size = min(batch_size, M - _)  # Adjust batch size for the last batch
        paths = np.zeros((current_batch_size, steps + 1))
        paths[:, 0] = V_0  # Initialize paths
        
        # Generate paths for the current batch
        for t in range(1, steps + 1):
            Z = np.random.randn(current_batch_size)  # Standard normal for each simulation
            paths[:, t] = paths[:, t-1] * np.exp((0 - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        # Final values at T for the current batch
        V_T = paths[:, -1]
        
        # Count defaults in the current batch
        total_defaulted += np.sum(V_T < K)
    
    # Calculate default probability
    prob_default = (total_defaulted / M) * 100
    print(f'The defaults with batched paths is: {total_defaulted}')
    
    return prob_default





# def credit_spread_model(V, K, sigma, r, T, t):
#     if K != 0:
#         # 1) Riskless bond price (for face value K)
#         riskless = np.exp(-r * (T - t))

#         # 2) Calculate d1 and d2
#         d1 = (np.log(V / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
#         d2 = d1 - sigma * np.sqrt(T - t)

#         # 3) *Defaultable* bond price using Merton's debt formula
#         defaultable_bond = K * riskless * norm.cdf(d2) + V * (1 - norm.cdf(d1))

#         # 4) Credit spread calculation: -1/(T - t) * ln(defaultable_bond / riskless_bond)
#         credit_spread = -1/(T - t) * np.log(defaultable_bond / (K * riskless))

#     else: 

#         credit_spread = 0

#     return credit_spread

# def credit_spread_model_old(V, K, sigma, r, T, t):
#     if K!= 0:
#         riskless = np.exp(-r * (T - t))  # Riskless bond price

#         # Calculate d1 and d2
#         d1 = (np.log(V / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
#         d2 = d1 - sigma * np.sqrt(T - t)

#         # Defaultable bond price using Merton's formula
#         defaultable_bond = V * norm.cdf(d1) - K * riskless * norm.cdf(d2)
#         print(defaultable_bond / ( K * riskless))

#         # Credit spread calculation
#         credit_spread = -1 / (T - t) * np.log(defaultable_bond / (K * riskless))

#         return credit_spread
    
#     else: 
#         return 0 
