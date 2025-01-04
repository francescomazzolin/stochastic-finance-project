import numpy as np 
from scipy.stats import norm


### Preparing the functions for the iterative procedure to compute asset values
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

def black_scholes_equation(V_t, S_t, B, r, T = 1, sigma_V = 0.1):
    """
    Compute the Black-Scholes equity price difference for an array of S_t, B, and r.
    This function is used to find the root for V_t.
    
    Parameters:
    V_t : float
        Asset value to solve for (scalar).
    S_t, B, r : np.array
        Arrays of equity prices, debt face values, and risk-free rates.
    T, sigma_V : float
        Constants: time to maturity and asset volatility.
    t : float
        Current time.

    Returns:
    np.array
        Difference between the calculated equity price and observed equity price S_t.
    """
    # Calculate d_t1 and d_t2 for all elements in S_t, B, and r
    
    d_t1 = (np.log(V_t / B) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d_t2 = d_t1 - sigma_V * np.sqrt(T)
    
    # Black-Scholes formula for equity price
    equity_price = V_t * norm.cdf(d_t1) - B * np.exp(-r * T) * norm.cdf(d_t2)
    
    # Return the difference from observed S_t
    return equity_price - S_t

def invert_black_scholes(S_t, B, r, T=1, sigma_V = 0.1, initial_guess=100):
    """
    Invert the Black-Scholes formula to estimate V_t for arrays of S_t, B, and r.
    
    Parameters:
    S_t, B, r : np.array
        Arrays of observed equity prices, debt face values, and risk-free rates.
    T, sigma_V : float
        Constants: time to maturity and asset volatility.
    t : float
        Current time.
    initial_guess : float
        Initial guess for V_t.

    Returns:
    np.array
        Estimated asset values (V_t) for each input set.
    """
    # Vectorized root finding for each element in S_t, B, and r
    V_t_estimates = []
    for S_t_i, B_i, r_i in zip(S_t, B, r):
        # Initial bracket
        a, b = 1, 400
        f_a = black_scholes_equation(a, S_t_i, B_i, r_i, T, sigma_V)
        f_b = black_scholes_equation(b, S_t_i, B_i, r_i, T, sigma_V)
        
        # Dynamically adjust the bracket until signs are opposite
        while f_a * f_b > 0:
            a /= 2
            b *= 2
            f_a = black_scholes_equation(a, S_t_i, B_i, r_i, T, sigma_V)
            f_b = black_scholes_equation(b, S_t_i, B_i, r_i, T, sigma_V)
        
        # Solve for V_t using root_scalar
        solution = root_scalar(
            black_scholes_equation,
            args=(S_t_i, B_i, r_i, T, sigma_V),
            bracket=[a, b],
            method='brentq'
        )
        V_t_estimates.append(solution.root)
    
    return np.array(V_t_estimates)
"""
# Example Usage
S_t = np.array([50e8, 60e8, 70e8])  # Observed equity prices
B = np.array([100e6, 120e6, 150e6])  # Face values of debt
r = np.array([0.05, 0.04, 0.03])  # Risk-free rates
T = 1.0  # Maturity in years (constant)
t = 0     # Current time
sigma_V = 0.2  # Asset volatility (constant)

V_t_estimated = invert_black_scholes(S_t, B, r, T, sigma_V)
print(f"Estimated Asset Values (V_t): {V_t_estimated}")
"""
instrument = 'AAPL.O'

prices = data.loc[instrument]['Close Price']
debt = data.loc[instrument]['Debt - Total'] / 10e9
r = [0.03] * prices.shape[0] 
r = np.array(r)
log_returns = np.log(prices / prices.shift(1)).dropna()
volatility_guess = log_returns.std()

#print(volatility_guess)
#print(prices)
equity_values = (data.loc[instrument]['Shares used to calculate Diluted EPS - Total'] * data.loc[instrument]['Close Price']) / 10e9

value_guess = invert_black_scholes(equity_values, debt, r, sigma_V=volatility_guess)
print(value_guess * 10e9)
equity_values

# Function calling

"""
V = Total company value
K = Liability (strike price)
r = interest rate
sigma = standard deviation
T = Maturity
t = time (a volte non è nemmeno messo ho visto in alcune formule ma nel nostro libro c'è)


V = St+Bt

ST può essere vista come una call option ST = (VT-BT)+
BT come una Put option in pratica BT = B - (B-VT)+

"""

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
#%% Inputs
r = 0.1        # Risk-free rate (10%)
sigma = 0.2    # Asset volatility (10%)
T = 1          # Maturity time (1 year)
t = 0          # Current time (0 years)

# Compute Equity and Debt
#equity_value = merton_equity(V, K, r, sigma, T, t)
#debt_value = merton_debt(V, K, r, sigma, T, t)

# Validation: Sum of Equity and Debt should equal Total Assets
#validation = equity_value + debt_value

# Output Results
#print(f"Equity Value (S) is: {equity_value:.6f}")
#print(f"Debt Value (B) is: {debt_value:.6f}")
#print(f"Difference between (S+B) and V: {validation:.6f} vs {V}")