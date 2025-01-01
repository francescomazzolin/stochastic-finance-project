#%%
import eikon as ek
import pandas as pd
import numpy as np
# Set API Key
ek.set_app_key('86872e92eb6d46a0a1b182488b3c6bff38c6b468')


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
    df.drop(columns=['RIC'])

    # Add Total Value column
    df['Total_value'] = df['Common Equity - Total'] + df['Debt - Total']

    return df
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
    
    # Return results as a DataFrame
    return pd.DataFrame(results)

# Example usage
volatility = volatility_estimator(['AAPL.O', 'META.O'], start_date=None, end_date=None)
print(volatility)



    

