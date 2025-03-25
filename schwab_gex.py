import base64
import requests
import os
from dotenv import load_dotenv
import logging
import json
from math import exp, sqrt
import numpy as np
import matplotlib.pyplot as plt  # Add this import at the top

# Load environment variables
load_dotenv()

class SchwabMarketData:
    def __init__(self):
        self.base_url = "https://api.schwabapi.com/marketdata/v1"
        self.client_id = os.getenv('SCHWAB_CLIENT_ID')
        self.client_secret = os.getenv('SCHWAB_CLIENT_SECRET')
        self.access_token = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not self.client_id or not self.client_secret:
            self.logger.error("Missing API credentials")
            raise ValueError("API credentials not found")

    def get_auth_token(self):
        auth_url = "https://api.schwabapi.com/v1/oauth/token"
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode('ascii')
        base64_auth = base64.b64encode(auth_bytes).decode('ascii')
        
        headers = {
            "Authorization": f"Basic {base64_auth}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        try:
            response = requests.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            return True
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return False

    def get_spx_option_chain(self, strike_count=20):
        if not self.access_token and not self.get_auth_token():
            return None
                
        endpoint = f"{self.base_url}/chains"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }
        params = {
            "symbol": "$SPX",  # Correct symbol format for SPX
            "contractType": "ALL",
            "strikeCount": strike_count,
            "includeQuotes": "TRUE",
            "strategy": "SINGLE"  # Changed from ANALYTICAL to SINGLE
        }
        
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get SPX chain: {str(e)}")
            return None

    def calculate_gamma_by_strike(self, option_chain):
        """Calculate gamma exposure by strike price"""
        if not option_chain:
            self.logger.error("No option chain data provided")
            return None

        # Debug: Print sample of option chain data
        self.logger.info("Sample option chain data:")
        self.logger.info(json.dumps(option_chain, indent=2)[:1000])  # First 1000 chars

        # Get underlying price
        underlying_price = option_chain.get('underlyingPrice')
        if not underlying_price:
            self.logger.error("Missing underlying price in option chain")
            return None
            
        # Initialize dictionary to store gamma by strike
        gamma_by_strike = {}
        
        # Process calls and puts
        for option_type in ['callExpDateMap', 'putExpDateMap']:
            if option_type not in option_chain:
                self.logger.error(f"Missing {option_type} in option chain")
                continue
                
            for exp_date in option_chain[option_type]:
                for strike in option_chain[option_type][exp_date]:
                    option = option_chain[option_type][exp_date][strike][0]
                    
                    # Debug: Print option data
                    self.logger.debug(f"Processing {option_type} at {strike}: {option}")
                    
                    # Verify required fields
                    if not all(key in option for key in ['openInterest', 'delta', 'volatility', 'daysToExpiration']):
                        self.logger.warning(f"Missing required fields in option data for strike {strike}")
                        continue
                        
                    strike_price = float(strike)
                    open_interest = float(option['openInterest'])
                    gamma = float(option.get('gamma', 0))
                    
                    # Debug: Print raw values
                    self.logger.debug(f"Strike: {strike_price}, OI: {open_interest}, Gamma: {gamma}")
                    
                    # If gamma not provided, estimate it
                    if gamma == 0:
                        try:
                            S = float(underlying_price)
                            K = float(strike_price)
                            T = float(option['daysToExpiration']) / 365.0
                            vol = float(option['volatility']) / 100.0 if option['volatility'] != 'NaN' else 0.3
                            r = 0.02
                            
                            # Debug: Print calculation parameters
                            self.logger.debug(f"Calculation params: S={S}, K={K}, T={T}, vol={vol}")
                            
                            if T > 0 and vol > 0:
                                d1 = (np.log(S/K) + (r + vol**2/2)*T) / (vol * sqrt(T))
                                gamma = exp(-d1**2/2) / (S * vol * sqrt(2 * np.pi * T))
                                # Debug: Print calculated gamma
                                self.logger.debug(f"Calculated gamma: {gamma}")
                        except Exception as e:
                            self.logger.warning(f"Gamma calculation failed for strike {strike}: {str(e)}")
                            continue

                    # Calculate gamma exposure
                    gamma_exposure = gamma * open_interest * 100.0
                    # Debug: Print final gamma exposure
                    self.logger.debug(f"Gamma exposure: {gamma_exposure}")
                    
                    if strike_price in gamma_by_strike:
                        gamma_by_strike[strike_price] += gamma_exposure
                    else:
                        gamma_by_strike[strike_price] = gamma_exposure

        return gamma_by_strike

    def plot_gamma_exposure(self, gamma_data):
        """Plot gamma exposure by strike price"""
        if not gamma_data:
            self.logger.error("No gamma data to plot")
            return

        # Prepare data for plotting
        strikes = sorted(gamma_data.keys())
        exposures = [gamma_data[strike] for strike in strikes]

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(strikes, exposures, marker='o', linestyle='-')
        plt.title('SPX Gamma Exposure by Strike Price')
        plt.xlabel('Strike Price')
        plt.ylabel('Gamma Exposure')
        plt.grid(True)
        plt.tight_layout()
        
        # Show the plot
        plt.show()

def main():
    try:
        schwab = SchwabMarketData()
        
        # Get SPX option chain
        spx_chain = schwab.get_spx_option_chain(strike_count=20)
        if not spx_chain:
            print("Failed to retrieve SPX option chain")
            return

        # Calculate gamma by strike
        gamma_data = schwab.calculate_gamma_by_strike(spx_chain)
        
        if gamma_data:
            # Plot the gamma exposure
            schwab.plot_gamma_exposure(gamma_data)
        else:
            print("Failed to calculate gamma exposure")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()