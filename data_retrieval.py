import pandas as pd 
import eikon as ek 
import numpy as np
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
        print(data)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values(by='Date').reset_index(drop=True)

        data['Equity_Value'] = data['Close Price'] * data['Shares used to calculate Diluted EPS - Total']

        
        #Computing the log-returns
        prices = data['Close Price']
        log_returns = np.log(prices / prices.shift(1)).dropna()
        data['Log_Returns'] = log_returns
        data.dropna()

        #Compute the rolling-window estimation of the volatility using last year observations
        data['Rolling_Volatility'] = (data['Log_Returns'].rolling(window=rolling_window).std() * np.sqrt(rolling_window) )
        #data['Rolling_Volatility'] = data['Log_Returns'].std() * np.sqrt(252)
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












def single_company_bond_data(ric, fields_2, start_date):
    # Retrieve the bond data
        fields_1 = ['TR.BondISIN']
        bond_df, err = ek.get_data(ric, fields_1)

        #print(bond_df)

        bond_l = list(bond_df['Bond ISIN'])

        #print(bond_l[0])
        #print(type(bond_l[0]))

        #df, err = ek.get_data(bond_l, fields_2)

        try:

            df, err = ek.get_data(bond_l, fields_2, {'SDate': f'{start_date}'})

            df['RIC'] = ric
            return df

        except:
             
             return pd.DataFrame()

        
