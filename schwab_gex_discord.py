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
    snapshot_time = pd.to_datetime('now')
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

def _gamma_range(quote, from_range=0.8, to_range=1.2):
    spotPrice = quote['current_price']
    fromStrike = from_range * spotPrice
    toStrike = to_range * spotPrice
    return spotPrice, fromStrike, toStrike

def naive_gamma(quote, options):
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
        df.loc[df_.index, 'callGammaEx'] = _calcGammaExCall(level, df_.strike_price, df_.iv, df_.days_to_expiration, 0, 0, df_.open_interest)

        df_ = df[df.put_call == 'P']
        df.loc[df_.index, 'putGammaEx'] = _calcGammaExCall(level, df_.strike_price, df_.iv, df_.days_to_expiration, 0, 0, df_.open_interest)

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
    df = options.copy()

    groups = options.option.str.extract(r'([A-Z]+\d{6})[CP](\d+)')
    df['key'] = [f'{prefix}@{strike}' for (prefix, strike) in zip(groups[0], groups[1])]
    df_ = df[df.put_call == 'C']
    df.loc[df_.index, 'GEX'] = df_.gamma * df_.open_interest * 100 * spot * spot * 0.01
    df_ = df[df.put_call == 'P']
    df.loc[df_.index, 'GEX'] = df_.gamma * df_.open_interest * 100 * spot * spot * 0.01 * -1

    df = df[df.put_call == 'C'].merge(df[df.put_call == 'P'], on=['key', 'strike_price'], suffixes=['_call', '_put'])
    df['total_gamma'] = (df.GEX_call + df.GEX_put) / 10**9
    df_agg = df[['strike_price', 'GEX_call', 'GEX_put', 'total_gamma']].groupby('strike_price').sum().reset_index()

    return df_agg

def plot_gamma_exposure(todayDate, quote, levels, totalGamma, totalGammaExNext, totalGammaExFri, zeroGamma, nextExpiry, nextMonthlyExp):
    spotPrice, fromStrike, toStrike = _gamma_range(quote)
    fig, ax = plt.subplots(figsize=(24, 6))

    # Set background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    plt.grid(True, color='white')
    plt.plot(levels, totalGamma, label="All Expiries", color='white')
    if str(nextExpiry) != 'NaT':
        plt.plot(levels, totalGammaExNext, label=f"Ex-Next Expiry {nextExpiry.strftime('%d %b %Y')}", color='orange')
    if str(nextMonthlyExp) != 'NaT':
        plt.plot(levels, totalGammaExFri, label=f"Ex-Next Monthly Expiry {nextMonthlyExp.strftime('%d %b %Y')}", color='white')
    chartTitle = f"Gamma Exposure Profile, {quote['symbol']}, {todayDate.strftime('%d %b %Y')}"
    plt.title(chartTitle, fontweight="bold", fontsize=20, color='white')
    plt.xlabel('Index Price', fontweight="bold", color='white')
    plt.ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold", color='white')
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"{quote['symbol']} Spot: {spotPrice:,.0f}")
    
    # Handle zero gamma crossing point
    if zeroGamma is not None and np.isfinite(zeroGamma):
        plt.axvline(x=zeroGamma, color='cyan', lw=3, label=f"Gamma Flip: {zeroGamma:,.0f}")
        # Only fill between if we have valid zero gamma
        trans = ax.get_xaxis_transform()
        if np.isfinite(min(totalGamma)) and np.isfinite(max(totalGamma)):
            plt.fill_between([fromStrike, zeroGamma], min(totalGamma), max(totalGamma), 
                            facecolor='red', alpha=0.1, transform=trans)
            plt.fill_between([zeroGamma, toStrike], min(totalGamma), max(totalGamma), 
                            facecolor='green', alpha=0.1, transform=trans)
    else:
        plt.axvline(x=quote['current_price'], color='yellow', lw=2, linestyle='dotted', 
                    label='No Gamma Flip Detected')
    
    plt.axhline(y=0, color='grey', lw=1)
    plt.xlim([fromStrike, toStrike])
    plt.legend(facecolor=background_color, edgecolor='white', fontsize=12, 
              loc='upper left', framealpha=1, labelcolor='white')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=25))
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.show()

def plot_absoulte_gamma_exposure(quote, df_agg):
    spotPrice, fromStrike, toStrike = _gamma_range(quote)
    fig, ax = plt.subplots(figsize=(24, 6))

    # Set background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    plt.grid(True, color='white')

    # Determine colors for bars
    colors = ['#2da19a' if val > 0 else '#ef524f' for val in df_agg['total_gamma']]

    plt.bar(df_agg.strike_price, df_agg['total_gamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', color=colors)
    plt.xlim([fromStrike, toStrike])
    title = f"Total Gamma: ${df_agg.total_gamma.sum():.2f} Bn per 1% {quote['symbol']} Move"
    plt.title(title, fontweight="bold", fontsize=20, color='white')
    plt.xlabel('Strike', fontweight="bold", color='white')
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold", color='white')
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"{quote['symbol']} Spot - {spotPrice:,.0f}")
    plt.legend(facecolor=background_color, edgecolor='white', fontsize=12, loc='upper left', framealpha=1, labelcolor='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.show()

def plot_absoulte_gamma_exposure_by_calls_and_puts(quote, df_agg):
    spotPrice, fromStrike, toStrike = _gamma_range(quote)
    fig, ax = plt.subplots(figsize=(24, 6))

    # Set background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    plt.grid(True, color='white')
    plt.bar(df_agg.strike_price, df_agg['GEX_call'].to_numpy() / 10 ** 9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma", color='#2da19a')
    plt.bar(df_agg.strike_price, df_agg['GEX_put'].to_numpy() / 10 ** 9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma", color='#ef524f')
    plt.xlim([fromStrike, toStrike])
    title = f"Total Gamma: ${df_agg.total_gamma.sum():.2f} Bn per 1% {quote['symbol']} Move"
    plt.title(title, fontweight="bold", fontsize=20, color='white')
    plt.xlabel('Strike', fontweight="bold", color='white')
    plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="bold", color='white')
    plt.axvline(x=spotPrice, color='r', lw=1, label=f"{quote['symbol']} Spot - {spotPrice:,.0f}")
    plt.legend(facecolor=background_color, edgecolor='white', fontsize=12, loc='upper left', framealpha=1, labelcolor='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.show()

def plot_index_gamma_report(quote, options, snapshot_time):
    spot_price = quote['current_price']
    gamma_params = naive_gamma(quote, options)
    plot_gamma_exposure(snapshot_time, *gamma_params)
    df = spot_gamma(options, spot_price)
    plot_absoulte_gamma_exposure(quote, df)
    plot_absoulte_gamma_exposure_by_calls_and_puts(quote, df)

symbol = '$SPX'  # Correct Schwab symbol format

# Get data using Schwab API instead of CBOE
quote, options, snapshot_time = get_schwab_options_chain(symbol)
plot_index_gamma_report(quote, options, snapshot_time)

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

spot_price = quote['current_price']

# Calculate Gamma Exposure
spotprice_squared = spot_price * spot_price
dfTrimmedCall['CallGEX'] = dfTrimmedCall['gamma'] * (dfTrimmedCall['open_interest'] + dfTrimmedCall['volume']) * spotprice_squared
dfTrimmedPut['PutGEX'] = dfTrimmedPut['gamma'] * (dfTrimmedPut['open_interest'] + dfTrimmedPut['volume']) * spotprice_squared * -1

# Calculate TotalGamma
dfTrimmedCall['TotalGamma'] = dfTrimmedCall['CallGEX'] * spot_price / 10**11
dfTrimmedPut['TotalGamma'] = dfTrimmedPut['PutGEX'] * spot_price / 10**11

# Combine the data
dfCombined = pd.concat([dfTrimmedCall, dfTrimmedPut])

# Aggregate data by strikePrice
dfAgg = dfCombined.groupby(['strike_price']).sum()
strikes = dfAgg.index.values

Total = dfAgg['TotalGamma'].sum().round(0)

# Assume strikes and dfAgg['TotalGamma'] are already defined
# Find the index of the strike closest to the spot price
closest_strike_idx = np.argmin(np.abs(strikes - spot_price))

# Define the range: 30 strikes before and after the spot price
start_idx = max(0, closest_strike_idx - 30)
end_idx = min(len(strikes), closest_strike_idx + 30)

# Filter strikes and dfAgg['TotalGamma'] to this range
filtered_strikes = strikes[start_idx:end_idx]
filtered_gammas = dfAgg['TotalGamma'].to_numpy()[start_idx:end_idx]

# Plot Absolute Gamma Exposure
f, ax = plt.subplots(figsize=(13, 5), facecolor='#161a25')  # Set the size that you'd like (width, height)
plt.grid(color='#d6d9e0', linewidth=.2)
ax.set_facecolor('#161a25')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Update the title to include the current date and time
plt.xlabel('Strike', fontweight="bold", color='w')
plt.ylabel('GEX', fontweight="bold", color='w')
plt.axvline(x=spot_price, color='r', lw=2, linestyle='dashed', label="SPX Spot: " + str("{:,.0f}".format(spot_price)))

# Color bars based on gamma value
colors = ['#2da19a' if e > 0 else '#ef524f' for e in filtered_gammas]

# Update ticks and labels
plt.xticks(rotation=45)  # Rotates X-Axis Ticks by 45-degrees
plt.xticks(filtered_strikes, color='w')
plt.yticks(color='w')

# Plot the bar chart with the filtered strikes and gammas
plt.bar(filtered_strikes, filtered_gammas, width=4.5, linewidth=0.1, edgecolor='k', color=colors)
plt.legend(facecolor=background_color, edgecolor='white', fontsize=12, loc='upper left', framealpha=1, labelcolor='white')

# Save the figure with high resolution
f.savefig("myplot.png", dpi=2000, bbox_inches='tight')

# Show the plot
plt.show()
