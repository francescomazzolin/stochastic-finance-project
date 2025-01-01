import numpy as np
from scipy.stats import norm
"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE EQUITY VALUE BASED ON MERTON
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def merton_equity(V,K,r,sigma,T,t):

    d1 = (np.log(V/K) + (r + 0.5*sigma**2)*(T-t))/ (sigma * np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    St = V*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2)

    return St

"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE DEBT VALUE BASED ON MERTON
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def merton_debt(V,K,r,sigma,T,t):

    d1 = (np.log(V/K) + (r + 0.5*sigma**2)*(T-t))/ (sigma * np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    Bt = K*np.exp(-r*(T-t))*norm.cdf(d2) + V*(1-norm.cdf(d1))
    
    return Bt

"""'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE PROBABILY OF DEFAULT
"""''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def risk_neutral_default_probability(V, B, r, sigma, T, t):
    d2 = (np.log(V / B) + (r - 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    return norm.cdf(-d2)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
FUNCTION TO COMPUTE THE CREDIT SPREAD
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def credit_spread_model(V, K, sigma, r, T, t):
    riskless = np.exp(-r * (T - t))  # Riskless bond price

    # Calculate d1 and d2
    d1 = (np.log(V / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    # Defaultable bond price using Merton's formula
    defaultable_bond = V * norm.cdf(d1) - K * riskless * norm.cdf(d2)

    # Credit spread calculation
    credit_spread = -1 / (T - t) * np.log(defaultable_bond / (K * riskless))

    return credit_spread
