import requests
from datetime import datetime

def get_bitcoin_price_on_date(date_str):
    """
    Get the historical price of Bitcoin on a specific date using CoinGecko API.

    Args:
        date_str (str): Date in format 'dd-mm-yyyy'

    Returns:
        float: Bitcoin price in USD on the given date
    """
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/history"
    params = {
        'date': date_str,  # Format: dd-mm-yyyy
        'localization': 'false'
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        try:
            price = data['market_data']['current_price']['usd']
            print(f"Bitcoin price on {date_str} was: ${price}")
            return price
        except KeyError:
            print("Price data not available for that date.")
    else:
        print("Failed to fetch data:", response.status_code, response.text)

    return None

# Example usage:
if __name__ == "__main__":
    # Format must be dd-mm-yyyy
    get_bitcoin_price_on_date("30-06-2025")
