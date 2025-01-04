

#!pip install eikon



import eikon as ek

import pandas as pd

import numpy as np

from scipy.stats import norm

from scipy.optimize import fsolve

import analytical_functions as af

import importlib



importlib.reload(af)


# Setting up the API key

ek.set_app_key('6e2b6a629eb84c0b859dc5faf22d48f94d85de97')



fields = ['TR.CLOSEPRICE.date',

            'TR.CLOSEPRICE', 

            'TR.F.ShrUsedToCalcDilEPSTot','TR.F.MktCap', 'TR.F.DebtTot']



start_date = '2024-12-30'



end_date = '2018-12-30'



rolling_window = 252

#Setting up the function

def get_financial_data(rics, fields, 

                       start_date, end_date,

                       rolling_window):

    #rics = rics

    #Data fields we are interested in: 

      #Total value of equity

      #Total value of debt



    results = []



    df = pd.DataFrame()



    for ric in rics:

        data, err = ek.get_data([ric], fields,

                                {'SDate': f'{start_date}' , 'EDate': f'{end_date}', 'FRQ':'D','period':'FY0'}) #Not sure about period

        if err is None:

            data['RIC'] = ric

            results.append(data)

        else:

            print(f"Error retrieving data for {ric}: {err}")

        

        data['Date'] = pd.to_datetime(data['Date'])

        data = data.sort_values(by='Date').reset_index(drop=True)





        #Computing the log-returns

        prices = data['Close Price']

        log_returns = np.log(prices / prices.shift(1)).dropna()

        data['Log_Returns'] = log_returns

        data.dropna()



        #Compute the rolling-window estimation of the volatility using last year observations

        data['Rolling_Volatility'] = (data['Log_Returns'].rolling(window=rolling_window).std() * rolling_window) 

        data = data.dropna(subset=['Rolling_Volatility'])

        data = data.dropna(how = 'any')

        





        #Computing total asset values as a sum of the market capitalization + total debt

        #data['Total_value'] = data['Market Capitalization'] + data['Debt - Total']

        data['Total_value'] = (data['Close Price'] * data['Shares used to calculate Diluted EPS - Total']) + data['Debt - Total']

        

        

        data = data.set_index(['Instrument', 'Date'])



    #Combine results into a single DataFrame

        df = pd.concat([df, data])

    #df.drop(columns=['RIC'])



    #Add Total asset value column

    



    return df


rics_list = ['AAPL.O', 'META.O', 'BRBI.MI']

data = get_financial_data(rics_list, fields,

                          start_date, end_date,

                          252)

df = data.loc[rics_list[2]]



plt.figure()

plt.plot(df['Market Capitalization'] / df['Debt - Total'])

plt.show()

data

# import pickle

# import sys

# import os



# current_directory = os.getcwd()



# # Add the current working directory to sys.path

# if current_directory not in sys.path:

#     sys.path.append(current_directory)

#     print('Yes')





# with open('dataframe.pkl', mode= 'rb') as f:



#     data = pickle.load(f)

#Checks on single stocks

print(data.loc['AAPL.O'].head())

#Plot of outstanding shares 

import matplotlib.pyplot as plt



plt.figure()



plt.title('Number of shares through time')



plt.plot(data.loc['AAPL.O']['Shares used to calculate Diluted EPS - Total'])



plt.show()


# df_k = pd.DataFrame()



# for ric in rics_list:

#     df = data.loc[ric]

#     df = df.reset_index()

#     df['Total_value'] = (df['Close Price'] * df['Shares used to calculate Diluted EPS - Total']) +df['Debt - Total']

#     # Compute log returns and rolling volatility

#     df['Log_Returns'] = np.log(df['Close Price'] / df['Close Price'].shift(1))

#     df['Rolling_Volatility'] = df['Log_Returns'].rolling(window=252).std() 

#     df = df.dropna(subset=['Rolling_Volatility'])

    

#     # Compute total asset value (equity + debt)

#     #df['Total_value'] = df['Market Capitalization'] + df['Debt - Total']

#     df['RIC'] = ric

#     df = df.set_index([f'RIC', 'Date'])

#     df = df.dropna(how='any')



#     # Drop rows with missing data

    



#     df_k = pd.concat([df, df_k])



# data = df_k.copy()

data.loc[rics_list[1]].describe()

for ric in rics_list:



    plt.plot(data.loc[ric]['Rolling_Volatility'])

# Solve the system for each row

def solve_system(row):

    # Extract parameters from the row

    E = row['Close Price'] * row['Shares used to calculate Diluted EPS - Total'] # Equity (market capitalization)

    sigma_E = row['Rolling_Volatility']  # Rolling volatility as initial guess for sigma_A

    D = row['Debt - Total']          # Debt

    

    row['Risk_Free_Rate'] = 0.04

    r = row['Risk_Free_Rate']        # Assume a risk-free rate column

    

    T = 1                            # Time to maturity (1 year)

    t = 0                            # Current time



    # Define the system of equations

    def system(vars):

        if D > 0:

            A, sigma_A = vars  # Unknowns: Asset value and asset volatility

            equity_value = af.merton_equity(A, D, r, sigma_A, T, t)

            d1 = (np.log(A / D) + (r + 0.5 * sigma_A**2) * (T - t)) / (sigma_A * np.sqrt(T - t))

            f1 = equity_value - E  # Equation (2)

            f2 = (A / E) * norm.cdf(d1) * sigma_A - sigma_E  # Equation (4)



        else: 



            f1 = E 

            f2 = sigma_E

        return [f1, f2]



    # Initial guesses

    A_guess = row['Total_value']  # Total value (equity + debt)

    sigma_A_guess = row['Rolling_Volatility']  # Rolling volatility



    # Solve the system

    solution = fsolve(system, [A_guess, sigma_A_guess])

    return pd.Series({'A_solution': solution[0], 'sigma_A_solution': solution[1]})



# Apply the solver to the DataFrame

def solve_for_all(df):

    results = df.apply(solve_system, axis=1)

    return pd.concat([df, results], axis=1)


data.isna().sum()

df_with_solutions = solve_for_all(data)

df_with_solutions

x = df_with_solutions['Close Price'] * df_with_solutions['Shares used to calculate Diluted EPS - Total']



asset_value = df_with_solutions['A_solution']



result = asset_value - x

result.describe()

def compute_additional_metrics(row):

    # Extract inputs from the row

    """

    V = row['Total_value']  # Asset value

    sigma = row['Rolling_Volatility']  # Asset volatility

    """

    V = row['A_solution']

    sigma = row['sigma_A_solution']

    K = row['Debt - Total']  # Debt

    row['Risk_Free_Rate'] = 0.04

    r = row['Risk_Free_Rate']

    

    #print(V/K)



    T = 1  # Time to maturity

    t = 0  # Current time



    # Compute quantities using the provided functions

    equity_value = af.merton_equity(V, K, r, sigma, T, t)

    debt_value = af.merton_debt(V, K, r, sigma, T, t)

    default_probability = af.risk_neutral_default_probability(V, K, r, sigma, T, t)

    credit_spread = af.credit_spread_model(V, K, sigma, r, T, t)



    # Return results as a Series

    return pd.Series({

        'Merton_Equity_Value': equity_value,

        'Merton_Debt_Value': debt_value,

        'Default_Probability': default_probability,

        'Credit_Spread': credit_spread

    })

# Compute additional metrics for each row

df_with_metrics = df_with_solutions.apply(compute_additional_metrics, axis=1)



# Combine results with the original DataFrame

df_final = pd.concat([df_with_solutions, df_with_metrics], axis=1)

import matplotlib.pyplot as plt

for ric in rics_list:

    plt.figure()



    def_probab = df_final.loc[ric]['Default_Probability']

    plt.plot(def_probab)

    plt.title(f'{ric}')



    plt.grid()



    plt.show()

df_final.loc[rics_list[1]].tail()

print(df_final.loc[rics_list[0]]['Default_Probability'].describe())

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

#%%

data['Equity'] = data.apply(lambda row: merton_equity(row['Common Equity - Total'], row['Debt - Total'], r, sigma, T, t), axis=1)

data['Debt'] = data.apply(lambda row: merton_debt(row['Common Equity - Total'], row['Debt - Total'], r, sigma, T, t), axis=1)

