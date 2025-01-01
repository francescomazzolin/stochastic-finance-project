#%%
import pandas as pd
import numpy as np
from scipy.stats import norm
from data_retrieve import get_financial_data
from data_retrieve import volatility_estimator

data = get_financial_data(['AAPL.O','META.O'])
sigma = volatility_estimator(['AAPL.O','META.O'], start_date=None, end_date=None)
combined_df = pd.concat([data,sigma],axis=1)
combined_df.drop(columns=['RIC'], inplace=True)
print(combined_df)
#%%

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
#Inputs
r = 0.04   # Risk-free rate (10%)
#sigma = 0.2    # Asset volatility (10%)
T = 1          # Maturity time (1 year)
t = 0          # Current time (0 years)


combined_df['Equity'] = combined_df.apply(lambda row: merton_equity(row['Total_value'], row['Debt - Total'], r, row['Volatility'], T, t), axis=1)
combined_df['Debt'] = combined_df.apply(lambda row: merton_debt(row['Total_value'], row['Debt - Total'], r,  row['Volatility'], T, t), axis=1)

# Display the DataFrame
print(combined_df)

def risk_neutral_default_probability(V, B, r, sigma, T, t):
    d2 = (np.log(V / B) + (r - 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    return norm.cdf(-d2)

combined_df['Default_Probability'] = combined_df.apply(lambda row: risk_neutral_default_probability(row['Total_value'], row['Debt - Total'], r, row['Volatility'], T, t), axis=1)

print(combined_df)