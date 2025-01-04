import numpy as np
from scipy.stats import norm
from decimal import Decimal, getcontext

"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE EQUITY VALUE BASED ON MERTON
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def merton_equity(V,K,r,sigma,T,t):
    if K != 0:
        d1 = (np.log(V/K) + (r + 0.5*sigma**2)*(T-t))/ (sigma * np.sqrt(T-t))
        d2 = d1 - sigma*np.sqrt(T-t)
        St = V*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2)

    else:

        St = V

    return St

"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE DEBT VALUE BASED ON MERTON
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def merton_debt(V,K,r,sigma,T,t):
    if K != 0: 
        d1 = (np.log(V/K) + (r + 0.5*sigma**2)*(T-t))/ (sigma * np.sqrt(T-t))
        d2 = d1 - sigma*np.sqrt(T-t)
        Bt = K*np.exp(-r*(T-t))*norm.cdf(d2) + V*(1-norm.cdf(d1))

    else:

        Bt = 0
    
    return Bt

"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE PROBABILY OF DEFAULT
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def risk_neutral_default_probability(V, B, r, sigma, T, t, precision = 20):
    getcontext().prec = precision
    
    if B != 0:
        V = Decimal(V)
        B = Decimal(B)
        r = Decimal(r)
        sigma = Decimal(sigma)
        T = Decimal(T)
        #t = Decimal(t)
        d2 = (np.log(float(V / B)) + float((r - Decimal('0.5') * sigma**2) * (T - t))) / float(sigma * (T - t).sqrt())
        return float(norm.cdf(-d2))
    
    else:
        return 0
    
def default_probability(V, B, r, sigma, T, t, precision = 20):
    getcontext().prec = precision
    
    if B != 0:
        V_d = Decimal(V)
        B_d = Decimal(B)
        r_d = Decimal(r)
        sigma_d = Decimal(sigma)
        T_d = Decimal(T)
        #t = Decimal(t)
        arg = ( np.log(float(B_d/V_d)) - float(0 - Decimal('0.5') *sigma_d**2) ) / float(sigma_d*np.sqrt(T_d-t))
        return float(norm.cdf(arg))
    
    else:
        return 0



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


def monte_carlo_merton_anti(V_0, K, r, sigma, T, M):

    V_d = Decimal(V_0)
    K_d = Decimal(K)
    r_d = Decimal(r)
    sigma_d = Decimal(sigma)
    T_d = Decimal(T)
    t_d = Decimal(0)

    threshold = (np.log(float(K_d/V_d)) - float(0 - Decimal('0.5') *sigma_d**2)) / float(sigma_d*np.sqrt(T_d-t_d))
    print(f'The threshold is: {threshold}\n')

    threshold = (np.log(K / V_0) - (-0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    print(f'The threshold without the decimals is: {threshold}')

    W_T = np.random.randn(M) * np.sqrt(T)
    print(f'The number of observations below the threshold is: {np.sum(W_T < threshold)}')

    log_V_T = np.log(V_0) + ((-0.5 * sigma**2) * T) + (sigma * W_T)

    log_V_T_anti = np.log(V_0) + ((-0.5 * sigma**2) * T) - (sigma * W_T)

    total_sims = np.concatenate((log_V_T, log_V_T_anti))

    defaulted_simulations = np.sum(log_V_T < np.log(K))
    print(f'The number of simulated defaults is: {defaulted_simulations}')
    
    # Debug info
    print(f"Min log(V_T) - Anti: {np.min(total_sims)}, Max log(V_T) - Anti: {np.max(total_sims)}, Threshold - Anti: {np.log(K)}")
    prob_default = (defaulted_simulations / M)


    num_decimal = np.log(float(K_d / V_d))
    den_decimal = float(sigma_d * np.sqrt(T_d - t_d))
    term_decimal = float(Decimal('0.5') * sigma_d**2) / den_decimal
    threshold_decimal = num_decimal - term_decimal

    print(f"Decimal Calculation -> num: {num_decimal}, den: {den_decimal}, term: {term_decimal}, threshold: {threshold_decimal}")


    num_float = np.log(K / V_0)
    den_float = sigma * np.sqrt(T)
    term_float = (-0.5 * sigma**2) * T / den_float
    threshold_float = (num_float - term_float) / den_float

    print(f"Float Calculation -> num: {num_float}, den: {den_float}, term: {term_float}, threshold: {threshold_float}")

    
    return prob_default


def monte_carlo_path(V_0, K, r, sigma, T, M, steps):
    """
    Monte Carlo simulation for Merton model with daily paths.
    
    Parameters:
        V_0: float - Initial firm value.
        K: float - Debt face value (strike).
        r: float - Risk-free rate (unused in real-world drift case).
        sigma: float - Annualized volatility of firm value.
        T: float - Time to maturity in years.
        M: int - Number of simulations.
        
    Returns:
        prob_default: float - Default probability (%).
    """
    steps = int(252 * T)  # Daily steps
    dt = T / steps  # Time increment
    
    # Initialize paths
    paths = np.zeros((M, steps + 1))
    paths[:, 0] = V_0  # Start with V_0
    
    # Generate paths
    for t in range(1, steps + 1):
        Z = np.random.randn(M)  # Standard normal for each simulation
        paths[:, t] = paths[:, t-1] * np.exp((0 - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Final values at T
    V_T = paths[:, -1]
    
    # Count defaults at maturity
    defaulted_simulations = np.sum(V_T < K)
    print(f'The defaults with paths is: {defaulted_simulations}')
    prob_default = (defaulted_simulations / M) * 100
    
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






'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE CREDIT SPREAD
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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


def credit_spread_model(V, K, sigma, r, T, t):
    if K != 0:
        # 1) Riskless bond price (for face value K)
        p_0 = np.exp(-r * (T - t))

        # 2) Calculate d1 and d2
        d1 = (-(np.log((K*p_0)/V)) + (0 + 0.5*sigma**2) * (T - t)) / (sigma * np.sqrt(T -t))

        d2 = d1 - (sigma * np.sqrt(T - t))

        # 3) *Defaultable* bond price using Merton's debt formula
        #defaultable_bond = K * riskless * norm.cdf(d2) + V * (1 - norm.cdf(d1))

        # 4) Credit spread calculation: -1/(T - t) * ln(defaultable_bond / riskless_bond)
        credit_spread = -1/(T - t) * np.log(norm.cdf(d2) + (V/(K *p_0))*norm.cdf(-d1))

    else: 

        credit_spread = 0

    return credit_spread

