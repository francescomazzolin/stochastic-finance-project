#%%
import numpy as np
import eikon as ek
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("EIKON_API_KEY")
ek.set_app_key(api_key)
def jump_dev_estimator(rics, threshold,start_date=None, end_date=None):
    
    results = [] 

    for ric in rics:
            data = ek.get_timeseries(
                rics = ric,
                fields=["CLOSE"],
                start_date="1980-01-01",
                end_date=None,
                interval="daily"
            )
            if data is not None and not data.empty:
                data['log_returns'] = np.log(data['CLOSE']/data['CLOSE'].shift(1))
                mean_returns = np.mean(data['log_returns'])
                volatility = np.std(data['log_returns'])
                jumps = data[(data['log_returns'] - mean_returns).abs() >  threshold * volatility]
                if not jumps.empty:
                   v = jumps['log_returns'].std()  # Standard deviation of detected jumps
                else:
                   v = 0  # No detected jumps
                results.append({'RIC':ric, 'Jump_Std_Dev_v': v})
                
                print(f"RIC: {ric}")
                print(f"Mean Returns: {mean_returns:.4f}, Volatility: {volatility:.4f}")
                print(f"Number of jumps detected: {len(jumps)}")
                
    data = pd.DataFrame(results)    
            
    return data


jump_dev_estimator(['AAPL.O', 'META.O'],2.5 ,start_date=None, end_date=None)
