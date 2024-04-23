import os

import pandas as pd
import requests
from pandas.tseries.offsets import MonthEnd

# Fetch API key from environment variables
apikey = os.getenv('ALPHAVANTAGE_API_KEY')

if not apikey:
    raise EnvironmentError("API key not set in environment variables.")

# Function to fetch data
def fetch_data(url, params):
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch data:", response.status_code)
        return None

# URLs and parameters
base_url = f"https://www.alphavantage.co/query?apikey={apikey}"
stock_params = {
    "function": "TIME_SERIES_MONTHLY",
    "symbol": "WMT",
}
retail_params = {
    "function": "RETAIL_SALES",
}
commodities_params = {
    "function": "ALL_COMMODITIES",
    "interval": "monthly",
}
cpi_params = {
    "function": "CPI",
    "interval": "monthly",
}
unemployment_params = {
    "function": "UNEMPLOYMENT",
}

# Fetch data
stock_data = fetch_data(base_url, stock_params)
retail_data = fetch_data(base_url, retail_params)
commodities_data = fetch_data(base_url, commodities_params)
cpi_data = fetch_data(base_url, cpi_params)
unemployment_data = fetch_data(base_url, unemployment_params)

# Process stock data
if stock_data:
    monthly_data = stock_data["Monthly Time Series"]
    df_stock = pd.DataFrame.from_dict(monthly_data, orient='index')
    df_stock = df_stock[['4. close', '5. volume']]  # Select only the 'close' and 'volume' columns
    df_stock.columns = ['close', 'volume']
    df_stock.index = pd.to_datetime(df_stock.index).strftime('%Y-%m')
    df_stock.reset_index(inplace=True)
    df_stock.rename(columns={'index': 'date'}, inplace=True)
    df_stock = df_stock.astype({'close': 'float', 'volume': 'int'})

# Process retail sales data
if retail_data:
    df_retail = pd.DataFrame(retail_data['data'])
    df_retail['date'] = pd.to_datetime(df_retail['date']) - MonthEnd(1)
    df_retail['date'] = df_retail['date'].dt.strftime('%Y-%m')
    df_retail.rename(columns={'value': 'retail_sales'}, inplace=True)
    df_retail['retail_sales'] = pd.to_numeric(df_retail['retail_sales'], errors='coerce').astype('int')

# Process commodities data
if commodities_data:
    df_commodities = pd.DataFrame(commodities_data['data'])
    df_commodities['date'] = pd.to_datetime(df_commodities['date']) - MonthEnd(1)
    df_commodities['date'] = df_commodities['date'].dt.strftime('%Y-%m')
    df_commodities.rename(columns={'value': 'commodity_index'}, inplace=True)
    df_commodities['commodity_index'] = pd.to_numeric(df_commodities['commodity_index'], errors='coerce')

# Process CPI data
if cpi_data:
    df_cpi = pd.DataFrame(cpi_data['data'])
    df_cpi['date'] = pd.to_datetime(df_cpi['date']) - MonthEnd(1)
    df_cpi['date'] = df_cpi['date'].dt.strftime('%Y-%m')
    df_cpi.rename(columns={'value': 'cpi'}, inplace=True)
    df_cpi['cpi'] = pd.to_numeric(df_cpi['cpi'], errors='coerce')

# Process Unemployment data
if unemployment_data:
    df_unemployment = pd.DataFrame(unemployment_data['data'])
    df_unemployment['date'] = pd.to_datetime(df_unemployment['date']) - MonthEnd(1)
    df_unemployment['date'] = df_unemployment['date'].dt.strftime('%Y-%m')
    df_unemployment.rename(columns={'value': 'unemployment_rate'}, inplace=True)
    df_unemployment['unemployment_rate'] = pd.to_numeric(df_unemployment['unemployment_rate'], errors='coerce')

# Merge dataframes on the 'date' column using inner joins to ensure only shared dates are included
df_list = [df for df in [df_stock, df_retail, df_commodities, df_cpi, df_unemployment] if df is not None]
df_merged = df_list[0]
for df in df_list[1:]:
    df_merged = pd.merge(df_merged, df, on='date', how='inner')

# Drop rows with any NaN values
df_merged.dropna(inplace=True)

# Create data directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data", "full")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Save dataframe to a csv file in the data directory
output_file = os.path.join(data_dir, "full_dataset.csv")
df_merged.to_csv(output_file, index=False)
print("Data saved to", output_file)
print(df_merged)
