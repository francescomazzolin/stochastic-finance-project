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
        V = Decimal(V)
        B = Decimal(B)
        r = Decimal(r)
        sigma = Decimal(sigma)
        T = Decimal(T)
        #t = Decimal(t)
        arg = np.log(float(B/V)) - float(0 - Decimal('0.5') *sigma**2) / float(sigma*np.sqrt(T-t))
        return float(norm.cdf(arg))
    
    else:
        return 0



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