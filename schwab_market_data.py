import base64
import requests
from datetime import datetime
import json
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class SchwabMarketData:
    def __init__(self):
        self.base_url = "https://api.schwabapi.com/marketdata/v1"
        self.client_id = os.getenv('SCHWAB_CLIENT_ID')
        self.client_secret = os.getenv('SCHWAB_CLIENT_SECRET')
        self.access_token = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Verify credentials
        if not self.client_id or not self.client_secret:
            self.logger.error("Missing API credentials. Check your .env file.")
            raise ValueError("API credentials not found in .env file")

    def get_auth_token(self):
        """Get OAuth 2.0 access token using Basic Auth"""
        auth_url = "https://api.schwabapi.com/v1/oauth/token"
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('ascii')
        base64_auth = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            "Authorization": f"Basic {base64_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        data = {
            "grant_type": "client_credentials",
            "scope": "read_market_data"
        }
        
        try:
            response = requests.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            self.logger.info("Successfully obtained access token")
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            if response:
                self.logger.error(f"Response: {response.text}")
            return False

    def get_quotes(self, symbols):
        """Get quotes for multiple symbols"""
        if not self.access_token and not self.get_auth_token():
            self.logger.error("Failed to obtain access token")
            return None
                
        endpoint = f"{self.base_url}/quotes"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
        params = {
            "symbols": ",".join(symbols),
            "fields": "quote,reference"
        }
        
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get quotes: {str(e)}")
            if response:
                self.logger.error(f"Response: {response.text}")
            return None

    def get_option_chains(self, symbol, contract_type="ALL", strike_count=5):
        """Get option chains for a symbol"""
        if not self.access_token and not self.get_auth_token():
            return None
                
        endpoint = f"{self.base_url}/chains"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
        params = {
            "symbol": symbol,
            "contractType": contract_type,
            "strikeCount": strike_count,
            "includeQuotes": "TRUE"
        }
        
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get option chains: {str(e)}")
            if response:
                self.logger.error(f"Response: {response.text}")
            return None

# Test the implementation
if __name__ == "__main__":
    try:
        schwab = SchwabMarketData()
        
        # Test market quotes
        symbols = ["AAPL", "MSFT", "TSLA"]
        quotes = schwab.get_quotes(symbols)
        if quotes:
            print("\nMarket Quotes:")
            print(json.dumps(quotes, indent=2))
        else:
            print("Failed to get market quotes")
            
        # Test option chains
        options = schwab.get_option_chains("AAPL")
        if options:
            print("\nOption Chains:")
            print(json.dumps(options, indent=2))
        else:
            print("Failed to get option chains")
            
    except Exception as e:
        print(f"Error: {str(e)}")