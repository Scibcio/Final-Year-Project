import pandas as pd
import requests
from io import StringIO

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Fetch page
response = requests.get(url, headers=headers)

# Check if request worked
if response.status_code != 200:
    print("Failed to fetch page: ", response.status_code)
    exit()

# Convert HTML text into readable format for pandas
html_data = StringIO(response.text)

# Force parser
tables = pd.read_html(html_data, flavor="lxml")

# Extract first table
sp500_df = tables[0]

# Clean columns
sp500_df = sp500_df[['Security', 'Symbol']]
sp500_df.columns = ['Company', 'Ticker']

# Save CSV
sp500_df.to_csv('sp500_tickers.csv', index=False)

print("CSV saved successfully")