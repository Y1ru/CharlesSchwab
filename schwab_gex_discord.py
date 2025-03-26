import json
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import norm
from app import SchwabMarketDataClient  # Import the Schwab client from your app.py

background_color = '#161a25'

def get_schwab_options_chain(symbol):
    # Initialize Schwab client
    client = SchwabMarketDataClient()
    
    # Get option chain data
    # Get option chain data with retries
    max_retries = 3
    option_data = None
    for attempt in range(max_retries):
        try:
            option_data = client.get_option_chain(symbol=symbol, strike_count=30)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt+1} failed, retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"All {max_retries} attempts failed")
                raise

    if not option_data:
        raise ValueError("Failed to retrieve option chain data from Schwab API")
    
    if not option_data:
        raise ValueError("Failed to retrieve option chain data from Schwab API")
    
    # Process the quote data
    quote = {
        'symbol': symbol,
        'current_price': option_data['underlyingPrice']
    }
    
    # Process options data
    options_list = []
    for expiry in option_data['callExpDateMap']:
        for strike in option_data['callExpDateMap'][expiry]:
            for contract in option_data['callExpDateMap'][expiry][strike]:
                contract['put_call'] = 'C'
                options_list.append(contract)
                
        for strike in option_data['putExpDateMap'][expiry]:
            for contract in option_data['putExpDateMap'][expiry][strike]:
                contract['put_call'] = 'P'
                options_list.append(contract)
    
    options = pd.DataFrame(options_list)
    
    # Convert columns to match expected format
    options['expiration_date'] = pd.to_datetime(options['expirationDate'])
    options['strike_price'] = options['strikePrice']
    options['open_interest'] = options['openInterest']
    options['iv'] = options['volatility'] / 100  # Convert from percentage to decimal
    options['gamma'] = options['gamma']
    options['volume'] = options['totalVolume']
    
    # Create option symbol for consistency with original code
    options['option'] = options.apply(lambda x: f"{symbol}{x['expiration_date'].strftime('%y%m%d')}{x['put_call']}{int(x['strike_price']*1000)}", axis=1)
    
    # Calculate days to expiration
    snapshot_time = pd.Timestamp.now()  # Changed from pd.to_datetime('now')
    options['days_to_expiration'] = np.busday_count(
        pd.Series(snapshot_time).dt.date.values.astype('datetime64[D]'),
        options['expiration_date'].dt.date.values.astype('datetime64[D]')) / 262
    
    return quote, options, snapshot_time

def get_gamma_pivot(chain, field, sorted=True):
    chain = chain[[field, 'total_gamma']].groupby(field).sum()
    chain = chain.reset_index()
    if sorted:
        chain = chain.sort_values(by=field).reset_index(drop=True)
    return chain

# Black-Scholes European-Options Gamma
def _calcGammaExCall(S, K, iv, T, r, q, OI):
    d1 = (np.log(S / K) + T * (r - q + 0.5 * iv ** 2)) / (iv * np.sqrt(T))
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * iv * np.sqrt(T))
    return OI * 100 * S * S * 0.01 * gamma

def _isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

def _gamma_range(quote, from_range=0.98, to_range=1.02): 
    spotPrice = quote['current_price']
    fromStrike = from_range * spotPrice
    toStrike = to_range * spotPrice
    return spotPrice, fromStrike, toStrike

def naive_gamma(quote, options, r=0.043):
    spotPrice, fromStrike, toStrike = _gamma_range(quote)

    levels = np.linspace(fromStrike, toStrike, 60)

    options.loc[options['days_to_expiration'] <= 0, 'days_to_expiration'] = 1/262

    nextExpiry = options.expiration_date.min()
    options['isThirdFriday'] = options.expiration_date.apply(_isThirdFriday)
    thirdFridays = options.loc[options.isThirdFriday]
    nextMonthlyExp = thirdFridays.expiration_date.min()

    totalGamma = []
    totalGammaExNext = []
    totalGammaExFri = []

    # For each spot level, calc gamma exposure at that point
    df = options.copy()
    for level in levels:
        df_ = df[df.put_call == 'C']
        df.loc[df_.index, 'callGammaEx'] = _calcGammaExCall(level, df_.strike_price, df_.iv, df_.days_to_expiration, r, 0, df_.open_interest)

        df_ = df[df.put_call == 'P']
        df.loc[df_.index, 'putGammaEx'] = _calcGammaExCall(level, df_.strike_price, df_.iv, df_.days_to_expiration, r, 0, df_.open_interest)

        totalGamma.append(df.callGammaEx.sum() - df.putGammaEx.sum())

        exNxt = df.loc[df.expiration_date != nextExpiry]
        totalGammaExNext.append(exNxt.callGammaEx.sum() - exNxt.putGammaEx.sum())

        exFri = df.loc[df.expiration_date != nextMonthlyExp]
        totalGammaExFri.append(exFri.callGammaEx.sum() - exFri.putGammaEx.sum())

    totalGamma = np.array(totalGamma)
    totalGammaExNext = np.array(totalGammaExNext) / 10 ** 9
    totalGammaExFri = np.array(totalGammaExFri) / 10 ** 9

    if totalGamma.max() > 10 ** 9:
       totalGamma = totalGamma/ 10 ** 9

    if totalGammaExNext.max() > 10 ** 9:
       totalGammaExNext = totalGammaExNext/ 10 ** 9

    if totalGammaExFri.max() > 10 ** 9:
       totalGammaExFri = totalGammaExFri/ 10 ** 9

    # Find Gamma Flip Point
    zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]

    # Handle case where no zero crossing points are found
    if len(zeroCrossIdx) == 0:
        zeroGamma = None
    else:
        negGamma = totalGamma[zeroCrossIdx]
        posGamma = totalGamma[zeroCrossIdx + 1]
        negStrike = levels[zeroCrossIdx]
        posStrike = levels[zeroCrossIdx + 1]
        zeroGamma = posStrike - ((posStrike - negStrike) * posGamma / (posGamma - negGamma))
        zeroGamma = zeroGamma[0]

    return quote, levels, totalGamma, totalGammaExNext, totalGammaExFri, zeroGamma, nextExpiry, nextMonthlyExp

def spot_gamma(df, spot):
    df = df.copy()

    groups = df.option.str.extract(r'([A-Z]+\d{6})[CP](\d+)')
    df['key'] = [f'{prefix}@{strike}' for (prefix, strike) in zip(groups[0], groups[1])]
    df_ = df[df.put_call == 'C']
    df.loc[df_.index, 'GEX'] = df_.gamma * df_.open_interest * 100 * spot * spot * 0.01
    df_ = df[df.put_call == 'P']
    df.loc[df_.index, 'GEX'] = df_.gamma * df_.open_interest * 100 * spot * spot * 0.01 * -1

    df = df[df.put_call == 'C'].merge(df[df.put_call == 'P'], on=['key', 'strike_price'], suffixes=['_call', '_put'])
    df['total_gamma'] = (df.GEX_call + df.GEX_put) / 10**9
    df_agg = df[['strike_price', 'GEX_call', 'GEX_put', 'total_gamma']].groupby('strike_price').sum().reset_index()

    return df_agg

def plot_gamma_exposure(todayDate, quote, levels, totalGamma, totalGammaExNext, totalGammaExFri, zeroGamma, nextExpiry, nextMonthlyExp, ax=None):
    spotPrice, fromStrike, toStrike = _gamma_range(quote)
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 6))
        fig.patch.set_facecolor(background_color)
        show_plot = True
    else:
        show_plot = False
        
    # Set background color
    ax.set_facecolor(background_color)

    ax.grid(True, color='white')
    ax.plot(levels, totalGamma, label="All Expiries", color='white')
    if str(nextExpiry) != 'NaT':
        ax.plot(levels, totalGammaExNext, label=f"Ex-Next Expiry {nextExpiry.strftime('%d %b %Y')}", color='orange')
    if str(nextMonthlyExp) != 'NaT':
        ax.plot(levels, totalGammaExFri, label=f"Ex-Next Monthly Expiry {nextMonthlyExp.strftime('%d %b %Y')}", color='white')
    chartTitle = f"Gamma Exposure Profile, {quote['symbol']}, {todayDate.strftime('%d %b %Y')}"
    ax.set_title(chartTitle, fontweight="bold", fontsize=14, color='white')
    ax.set_xlabel('Index Price', fontweight="bold", color='white')
    ax.set_ylabel('Gamma Exposure ($\$ billions/1% move)', fontweight="bold", color='white')
    ax.axvline(x=spotPrice, color='r', lw=1, label=f"{quote['symbol']} Spot: {spotPrice:,.0f}")
    
    # Handle zero gamma crossing point
    if zeroGamma is not None and np.isfinite(zeroGamma):
        ax.axvline(x=zeroGamma, color='cyan', lw=3, label=f"Gamma Flip: {zeroGamma:,.0f}")
        # Only fill between if we have valid zero gamma
        trans = ax.get_xaxis_transform()
        if np.isfinite(min(totalGamma)) and np.isfinite(max(totalGamma)):
            ax.fill_between([fromStrike, zeroGamma], min(totalGamma), max(totalGamma), 
                            facecolor='red', alpha=0.1, transform=trans)
            ax.fill_between([zeroGamma, toStrike], min(totalGamma), max(totalGamma), 
                            facecolor='green', alpha=0.1, transform=trans)
    else:
        ax.axvline(x=quote['current_price'], color='yellow', lw=2, linestyle='dotted', 
                    label='No Gamma Flip Detected')
    
    ax.axhline(y=0, color='grey', lw=1)
    ax.set_xlim([fromStrike, toStrike])
    ax.legend(facecolor=background_color, edgecolor='white', fontsize=10, 
              loc='upper left', framealpha=1, labelcolor='white')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=15))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    if show_plot:
        plt.show()
        
    return ax

def plot_absoulte_gamma_exposure(quote, df_agg, ax=None):
    spotPrice, fromStrike, toStrike = _gamma_range(quote)
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 6))
        fig.patch.set_facecolor(background_color)
        show_plot = True
    else:
        show_plot = False
        
    # Set background color
    ax.set_facecolor(background_color)

    ax.grid(True, color='white')

    # Determine colors for bars
    colors = ['#2da19a' if val > 0 else '#ef524f' for val in df_agg['total_gamma']]

    ax.bar(df_agg.strike_price, df_agg['total_gamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', color=colors)
    ax.set_xlim([fromStrike, toStrike])
    title = f"Total Gamma: \\${df_agg.total_gamma.sum():.2f} Bn per 1% {quote['symbol']} Move"
    ax.set_title(title, fontweight="bold", fontsize=14, color='white')
    ax.set_xlabel('Strike', fontweight="bold", color='white')
    ax.set_ylabel('Spot Gamma Exposure ($\$ billions/1% move)', fontweight="bold", color='white')
    ax.axvline(x=spotPrice, color='r', lw=1, label=f"{quote['symbol']} Spot - {spotPrice:,.0f}")
    ax.legend(facecolor=background_color, edgecolor='white', fontsize=10, loc='upper left', framealpha=1, labelcolor='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    if show_plot:
        plt.show()
        
    return ax

def plot_absoulte_gamma_exposure_by_calls_and_puts(quote, df_agg, ax=None):
    spotPrice, fromStrike, toStrike = _gamma_range(quote)
    
    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 6))
        fig.patch.set_facecolor(background_color)
        show_plot = True
    else:
        show_plot = False
        
    # Set background color
    ax.set_facecolor(background_color)

    ax.grid(True, color='white')
    ax.bar(df_agg.strike_price, df_agg['GEX_call'].to_numpy() / 10 ** 9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma", color='#2da19a')
    ax.bar(df_agg.strike_price, df_agg['GEX_put'].to_numpy() / 10 ** 9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma", color='#ef524f')
    ax.set_xlim([fromStrike, toStrike])
    title = f"Total Gamma: \\${df_agg.total_gamma.sum():.2f} Bn per 1% {quote['symbol']} Move"
    ax.set_title(title, fontweight="bold", fontsize=14, color='white')
    ax.set_xlabel('Strike', fontweight="bold", color='white')
    ax.set_ylabel('Spot Gamma Exposure ($\$ billions/1% move)', fontweight="bold", color='white')
    ax.axvline(x=spotPrice, color='r', lw=1, label=f"{quote['symbol']} Spot - {spotPrice:,.0f}")
    ax.legend(facecolor=background_color, edgecolor='white', fontsize=10, loc='upper left', framealpha=1, labelcolor='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    if show_plot:
        plt.show()
        
    return ax

import datetime  # Add this at the top with other imports
import matplotlib.animation as animation

def plot_index_gamma_report(quote, options, snapshot_time, symbol, dashboard=True):  # Added symbol parameter
    # Create initial dashboard
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(background_color)
    
    # Add clock to the figure
    clock_text = fig.text(0.95, 0.98, '', fontsize=12, color='white', ha='right')
    
    def update_clock(text_obj):
        text_obj.set_text(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    def update_plots(frame):
        try:
            # Clear the figure
            fig.clear()
            fig.patch.set_facecolor(background_color)
            
            # Re-add clock text after clearing
            nonlocal clock_text
            clock_text = fig.text(0.95, 0.98, '', fontsize=12, color='white', ha='right')
            update_clock(clock_text)
            
            # Add title with dynamic symbol
            fig.suptitle(f"{symbol} Gamma Exposure Dashboard - Live", 
                        fontsize=24, fontweight='bold', color='white')
            
            # Get fresh data with user's symbol
            quote, options, snapshot_time = get_schwab_options_chain(symbol)
            gamma_params = naive_gamma(quote, options)
            df = spot_gamma(options, quote['current_price'])
            
            # Create subplots
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            
            # Plot updated charts
            plot_gamma_exposure(snapshot_time, *gamma_params, ax=ax1)
            plot_absoulte_gamma_exposure(quote, df, ax=ax2)
            plot_absoulte_gamma_exposure_by_calls_and_puts(quote, df, ax=ax3)
            
            # Update fourth chart (rest of the code remains the same)
            options_copy = options.copy()
            # Create a DataFrame for calls
            dfCalls = options_copy[options_copy['put_call'] == 'C']
            # Create a DataFrame for puts
            dfPuts = options_copy[options_copy['put_call'] == 'P']
            
            dfTrimmedCall = dfCalls[['open_interest', 'gamma', 'volume', 'strike_price']]
            dfTrimmedPut = dfPuts[['open_interest', 'gamma', 'volume', 'strike_price']]
            
            # Convert data types
            dfTrimmedCall = dfTrimmedCall.astype({
                'open_interest': float,
                'gamma': float,
                'volume': float,
                'strike_price': float
            })
            
            dfTrimmedPut = dfTrimmedPut.astype({
                'open_interest': float,
                'gamma': float,
                'volume': float,
                'strike_price': float
            })
            
            # Calculate Gamma Exposure
            spotPrice = quote['current_price']  # Get spot price from quote
            spotprice_squared = spotPrice * spotPrice
            dfTrimmedCall['CallGEX'] = dfTrimmedCall['gamma'] * (dfTrimmedCall['open_interest'] + dfTrimmedCall['volume']) * spotprice_squared
            dfTrimmedPut['PutGEX'] = dfTrimmedPut['gamma'] * (dfTrimmedPut['open_interest'] + dfTrimmedPut['volume']) * spotprice_squared * -1
            
            # Calculate TotalGamma
            dfTrimmedCall['TotalGamma'] = dfTrimmedCall['CallGEX'] * spotPrice / 10**11
            dfTrimmedPut['TotalGamma'] = dfTrimmedPut['PutGEX'] * spotPrice / 10**11
            
            # Combine the data
            dfCombined = pd.concat([dfTrimmedCall, dfTrimmedPut])
            
            # Aggregate data by strikePrice
            dfAgg = dfCombined.groupby(['strike_price']).sum()
            strikes = dfAgg.index.values
            
            # Find the index of the strike closest to the spot price
            closest_strike_idx = np.argmin(np.abs(strikes - spotPrice))
            
            # In the fourth chart section, update the strike range:
            # Define the range: 15 strikes before and after the spot price (changed from 10)
            start_idx = max(0, closest_strike_idx - 15)
            end_idx = min(len(strikes), closest_strike_idx + 15)
            
            # Filter strikes and dfAgg['TotalGamma'] to this range
            filtered_strikes = strikes[start_idx:end_idx]
            filtered_gammas = dfAgg['TotalGamma'].to_numpy()[start_idx:end_idx]
            
            # Plot the filtered strikes chart
            ax4.set_facecolor(background_color)
            ax4.grid(color='#d6d9e0', linewidth=.2)
            ax4.spines['right'].set_visible(False)
            ax4.spines['top'].set_visible(False)
            ax4.spines['left'].set_visible(False)
            ax4.spines['bottom'].set_visible(False)
            
            ax4.set_xlabel('Strike', fontweight="bold", color='w')
            ax4.set_ylabel('GEX', fontweight="bold", color='w')
            ax4.axvline(x=spotPrice, color='r', lw=2, linestyle='dashed', label=f"SPX Spot: {spotPrice:,.0f}")
            
            # Color bars based on gamma value
            colors = ['#2da19a' if e > 0 else '#ef524f' for e in filtered_gammas]
            
            # Update ticks and labels
            ax4.set_xticks(filtered_strikes)
            ax4.set_xticklabels(filtered_strikes, rotation=45, color='w')
            ax4.tick_params(axis='y', colors='w')
            
            # Plot the bar chart with the filtered strikes and gammas
            ax4.bar(filtered_strikes, filtered_gammas, width=4.5, linewidth=0.1, edgecolor='k', color=colors)
            ax4.legend(facecolor=background_color, edgecolor='white', fontsize=10, loc='upper left', framealpha=1, labelcolor='white')
            ax4.set_title("Filtered Strike Range Gamma Exposure", fontweight="bold", fontsize=14, color='white')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.subplots_adjust(wspace=0.2, hspace=0.3)
            
        except Exception as e:
            print(f"Error in update: {e}")
    
    # Create initial plot
    update_plots(0)
    
    # Create animation with 1 second interval
    ani = animation.FuncAnimation(fig, update_plots, interval=1000)
    plt.show()

def main():
    # Get symbol input from user
    symbol = input("Enter symbol (e.g., $SPX, AAPL): ").strip().upper()
    
    # Add default to $SPX if no input
    if not symbol:
        symbol = '$SPX'
        print(f"No symbol entered, defaulting to {symbol}")
    
    # Add $ prefix for indices if not present
    if symbol in ['SPX', 'VIX', 'NDX', 'RUT'] and not symbol.startswith('$'):
        symbol = f'${symbol}'
        print(f"Added $ prefix for index: {symbol}")
    
    try:
        quote, options, snapshot_time = get_schwab_options_chain(symbol)
        plot_index_gamma_report(quote, options, snapshot_time, symbol, dashboard=True)  # Added symbol parameter
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")

if __name__ == "__main__":
    main()
