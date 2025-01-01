#%%
import pandas as pd
import numpy as np
from scipy.stats import norm
from IPython.display import display
from data_retrieve import get_financial_data
from data_retrieve import volatility_estimator
from monte_carlo_defaults import monte_carlo_merton
from analytical_functions import merton_debt,merton_equity,risk_neutral_default_probability

'''
IMPORTING THE DATA
'''
data = get_financial_data(['AAPL.O','META.O'])
sigma = volatility_estimator(['AAPL.O','META.O'], start_date=None, end_date=None)
combined_df = pd.concat([data,sigma],axis=1)
combined_df.drop(columns=['RIC'],inplace=True)

'''
INPUT FROM EXTERNAL RESOURCES
'''
r = 0.04       # Risk-free rate USA
T = 1          # Maturity time (1 year)
t = 0          # Current time (0 years)
M = 10000000   #Number of simulations

'''
COMPUTING THE VALUES FOR THE ANALYTICAL MODEL
'''
combined_df['Equity'] = combined_df.apply(lambda row: merton_equity(row['Total_value'], row['Debt - Total'], r, row['Volatility'], T, t), axis=1)
combined_df['Debt'] = combined_df.apply(lambda row: merton_debt(row['Total_value'], row['Debt - Total'], r,  row['Volatility'], T, t), axis=1)
combined_df['Default_Probability'] = combined_df.apply(lambda row: risk_neutral_default_probability(row['Total_value'], row['Debt - Total'], r, row['Volatility'], T, t), axis=1)

'''
MONTE CARLO RESULTS
'''
combined_df[['MC_Equity','MC_Dect','MC_Prob']] = combined_df.apply(lambda row: pd.Series(monte_carlo_merton(row['Total_value'], row['Debt - Total'],r,row['Volatility'],T,M)), axis=1)

'''
FINAL RESULTS
'''
combined_df



