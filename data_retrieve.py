#%%
import eikon as ek
import pandas as pd
import numpy as np
# Set API Key
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("EIKON_API_KEY")
ek.set_app_key(api_key)


# Function to retrieve data
def get_financial_data(rics):
    rics = rics
    fields = ['TR.F.ComEqTot', 'TR.F.DebtTot']
    results = []

    for ric in rics:
        data, err = ek.get_data(instruments=[ric], fields=fields)
        if err is None:
            data['RIC'] = ric
            results.append(data)
        else:
            print(f"Error retrieving data for {ric}: {err}")

    # Combine results into a single DataFrame
    df = pd.concat(results, ignore_index=True)
    

    # Add Total Value column
    df['Total_value'] = df['Common Equity - Total'] + df['Debt - Total']
    df.drop(columns=['RIC'],inplace=True)
    return df
get_financial_data(['AAPL.O', 'META.O'])
#%%

def volatility_estimator(rics, start_date=None, end_date=None):
    """
    Estimate the annualized historical equity volatility for the given RICs.
    
    Parameters:
        rics (list): List of RICs (e.g., ['AAPL.O', 'MSFT.O']).
        start_date (str): Start date in 'YYYY-MM-DD' format (optional).
        end_date (str): End date in 'YYYY-MM-DD' format (optional).
        
    Returns:
        pd.DataFrame: A DataFrame containing the annualized volatilities.
    """
    results = []
    for ric in rics:
        try:
            # Retrieve historical data
            data = ek.get_timeseries(
                rics = ric,
                fields=["CLOSE"],
                start_date=start_date,
                end_date=end_date,
                interval="daily"
            )
            if data is not None and not data.empty:
                # Compute log returns
                data['log_returns'] = np.log(data['CLOSE'] / data['CLOSE'].shift(1))
                # Calculate annualized volatility
                sigma_s = data['log_returns'].std() * np.sqrt(252)
                results.append({'RIC': ric, 'Volatility': sigma_s})
            else:
                print(f"No data retrieved for {ric}.")
        except Exception as e:
            print(f"Error retrieving data for {ric}: {e}")
    data = pd.DataFrame(results)
   
    data.drop(columns=['RIC'], inplace=True)
    
    
    # Return results as a DataFrame
    return data



    

