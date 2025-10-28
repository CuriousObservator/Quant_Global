#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 15:27:09 2025

@author: rbarcrosspbar
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-darkgrid')


def back_test_data(omega=0.000001, alpha=0.05, beta=0.94, 
    s0=100.0, daily_bars=25, start_date="2025-01-01", 
    dt=0.00015873, days=252, drift=0.01):
    start_date = pd.to_datetime(start_date)
    np.random.seed(43) 
    
    data = []
    s_t = s0
    current_date = start_date
    
    
    variance = omega / (1 - alpha - beta)
    last_return_sq = 0.0

    for day in range(days):
        market_time = current_date.replace(hour=9, minute=15, second=0)

        for bar in range(daily_bars):
            s_open = s_t
            market_time += pd.Timedelta(minutes=15)
            
            variance = omega + alpha * last_return_sq + beta * variance
            volatility = np.sqrt(variance)
            
            random_walk = np.random.normal(0, 1)
            
            price_change_pct = (drift * dt) + (volatility * random_walk * np.sqrt(dt))
            
        
            s_t *= (1 + price_change_pct)

            
            last_return_sq = price_change_pct ** 2
            
            
            s_t = max(0.01, s_t) 


            s_high = max(s_open, s_t)
            s_low = min(s_open, s_t)
            
            data.append({
                'Datetime': market_time,
                'Open': s_open,
                'High': s_high,
                'Low': s_low,
                'Close': s_t
            })
        
        current_date += pd.Timedelta(days=1)

    stock = pd.DataFrame(data)
    stock.set_index('Datetime', inplace=True)
    stock["Returns"] = stock["Close"].pct_change()
    
    return stock

stock = back_test_data(drift=0.001)

deviation = 2
stock["Middle_band"] = stock["Close"].rolling(window=40).mean()
stock["Stddev"] = stock["Close"].rolling(window=40).std()
stock["Upper_band"] = stock["Middle_band"] + deviation * stock["Stddev"]
stock["Lower_band"] = stock["Middle_band"] - deviation * stock["Stddev"]


stock["Raw_Buy_Signal"] = np.where(stock["Close"] < stock["Lower_band"], 1, 0)
stock["Raw_Sell_Signal"] = np.where(stock["Close"] > stock["Upper_band"], 1, 0)

stock["Buy"] = stock["Raw_Buy_Signal"].shift(1).fillna(0).astype(int)
stock["Sell"] = stock["Raw_Sell_Signal"].shift(1).fillna(0).astype(int)

capital = 100000.0 
initial_cap = capital
commission = 0.0001 
prev_signal = "hold"
inventory = 0 
trade_log = [] 
equity_over_time = []
stop_loss_pct = 0.015

total_trades = 0
total_round_trip_trades = 0
profitable_trades = 0
total_pnl_per_trade = 0.0

risk_per_trade = 10000.0 
trades = 0
sell_price = 0
buy_price = 0


for date, row in stock.iterrows():
     
    buy_signal = row["Buy"]
    sell_signal = row["Sell"]
    price = row["Open"]
    
    if inventory > 0:
        stop_level = buy_price * (1 - stop_loss_pct)
        if price <= stop_level:
            
            shares_held = abs(inventory)
            round_trip_comm = shares_held * commission * 2
            pnl = shares_held * (price - buy_price) - round_trip_comm
            
            total_round_trip_trades += 1
            total_pnl_per_trade += pnl
            if (pnl > 0): profitable_trades += 1 
            
            capital += pnl
            inventory = 0
            prev_signal = "hold"
            trades += 1
            
            trade_log.append({
                'ID': trades, 'Date': date, 'Action': 'STOP LOSS Long',
                'Shares': shares_held, 'Entry Price': buy_price, 'Exit Price': price,
                'Realized PnL': pnl, 'Position Type': 'SL Close',
            })
            continue 
            
    elif inventory < 0:
        stop_level = sell_price * (1 + stop_loss_pct)
        if price >= stop_level:
            
            shorts_held = abs(inventory)
            round_trip_comm = shorts_held * commission * 2
            pnl = shorts_held * (sell_price - price) - round_trip_comm
            
            total_round_trip_trades += 1
            total_pnl_per_trade += pnl
            if (pnl > 0): profitable_trades += 1
            
            capital += pnl
            inventory = 0
            prev_signal = "hold"
            trades += 1
            
            trade_log.append({
                'ID': trades, 'Date': date, 'Action': 'STOP LOSS Short',
                'Shares': shorts_held, 'Entry Price': sell_price, 'Exit Price': price,
                'Realized PnL': pnl, 'Position Type': 'SL Close',
            })
            continue 
    
    if buy_signal == 1 and prev_signal != "buy":
        
        if inventory < 0:
            trades += 1
            shorts_held = abs(inventory)
            round_trip_comm = shorts_held * commission * 2
            pnl = shorts_held * (sell_price - price) - round_trip_comm
            
            total_round_trip_trades += 1
            total_pnl_per_trade += pnl
            
            if(pnl>0):
                profitable_trades+=1
            
            capital += pnl
            inventory = 0
            
            trade_log.append({
                'ID': trades,
                'Date': date,
                'Action': 'Closed Shorts',
                'Shares': shorts_held,
                'Entry Price': sell_price,
                'Exit Price': price,
                'Realized PnL': pnl,
                'Position Type': 'Close',
            })
            
        
        if inventory == 0:
            trades += 1
            shares_bought = int(risk_per_trade/price)
            capital -= shares_bought*(price + commission)
            inventory += shares_bought
            buy_price = price
            prev_signal = "buy"
            trade_log.append({
                'ID': trades,
                'Date': date,
                'Action': 'Long',
                'Shares': shares_bought,
                'Entry Price': buy_price,
                'Exit Price': None,
                'Realized PnL': None,
                'Position Type': 'Long',
            })
        
        if inventory > 0:
            continue
        
    elif sell_signal == 1 and prev_signal != "sell":
    
        if inventory > 0:
            trades += 1
            shares_held = abs(inventory)
            round_trip_comm = shares_held * commission * 2
            pnl = shares_held * (price - buy_price) - round_trip_comm
            
            total_round_trip_trades += 1
            total_pnl_per_trade += pnl
            
            capital += pnl
            if(pnl>0):
                profitable_trades +=1
            
            inventory = 0
            prev_signal = "sell"
            trade_log.append({
                'ID': trades,
                'Date': date,
                'Action': 'Closed Long',
                'Shares': shares_held,
                'Entry Price': buy_price,
                'Exit Price': price,
                'Realized PnL': pnl,
                'Position Type': 'Close',
            })
            
        
        if inventory == 0:
            trades += 1
            shorts_sold = int(risk_per_trade/price)
            capital += shorts_sold*(price - commission)
            inventory -= shorts_sold
            sell_price = price
            prev_signal = "sell"
            trade_log.append({
                'ID': trades,
                'Date': date,
                'Action': 'Short',
                'Shares': shorts_sold,
                'Entry Price': sell_price,
                'Exit Price': None,
                'Realized PnL': None,
                'Position Type': 'Short',
            })
        
        if inventory < 0:
            continue      

    else:
        prev_signal = "hold"
    
    unrealized_pnl = 0.0
    if inventory != 0:
        if inventory > 0: 
            unrealized_pnl = (row["Close"] - buy_price) * inventory
        else: 
            unrealized_pnl = (sell_price - row["Close"]) * abs(inventory)

    
    total_equity = capital + unrealized_pnl

    equity_over_time.append({
        'Datetime': date,
        'Equity': total_equity,
        'Inventory': inventory,
        'Unrealized_PnL': unrealized_pnl
    })



equity_df = pd.DataFrame(equity_over_time)
equity_df.set_index('Datetime', inplace=True)

final_close_price = stock["Close"].iloc[-1]

final_unrealized_pnl = 0.0
if inventory != 0:
    if inventory > 0: 
        final_unrealized_pnl = (final_close_price - buy_price) * inventory
    else: 
        final_unrealized_pnl = (sell_price - final_close_price) * abs(inventory)

final_equity = capital + final_unrealized_pnl
total_pnl = final_equity - initial_cap


equity_series = equity_df['Equity']

equity_series.iloc[-1] = final_equity 


returns = equity_series.pct_change().dropna()

annualization_factor = np.sqrt(len(returns) * (252 * 25) / len(returns)) # Simplifies to sqrt(252*25) if returns covers the full period
annualization_factor = np.sqrt(252 * 25) 
RISK_FREE_RATE_DAILY = 0.02 / (252 * 25) 

excess_returns = returns - RISK_FREE_RATE_DAILY
sharpe_ratio = annualization_factor * (excess_returns.mean() / excess_returns.std())


rolling_max = equity_series.cummax()

drawdown = (equity_series / rolling_max) - 1
max_drawdown = drawdown.min()

win_ratio = profitable_trades / total_round_trip_trades if total_round_trip_trades > 0 else 0


print("="*50)
print("             BACKTEST PERFORMANCE METRICS (GARCH + BB)")
print("="*50)
print(f"Initial Capital:           ${100000.0:,.2f}") 
print(f"Final Equity:              ${final_equity:,.2f}")
print(f"Total PnL:                 ${total_pnl:,.2f}")
print(f"Total Return:              {(final_equity / 100000.0 - 1):.2%}")
print("-" * 50)
print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
print(f"Maximum Drawdown:          {max_drawdown:.2%}")
print(f"Total Trades Executed:     {trades:,}")
print(f"Total Trade Logs Recorded: {len(trade_log):,}")
print(f"Win Ratio (Uncalculated):  {win_ratio:.2%}") 
print("="*50)

#%%

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), sharex=True)

ax1 = axes[0]
ax1.plot(stock.index, stock['Close'], label='Close Price', color='lightblue')
ax1.plot(stock.index, stock['Upper_band'], label='Upper Band', color='orange', linestyle='--')
ax1.plot(stock.index, stock['Middle_band'], label='Middle Band (MA)', color='grey', linestyle='-')
ax1.plot(stock.index, stock['Lower_band'], label='Lower Band', color='orange', linestyle='--')


trade_df = pd.DataFrame(trade_log)


long_entries = trade_df[trade_df['Action'].str.contains('Long', na=False) & (trade_df['Position Type'] == 'Long')].set_index('Date')
ax1.scatter(long_entries.index, long_entries['Entry Price'], marker='^', color='green', label='Long Entry', alpha=0.7, s=50)

short_entries = trade_df[trade_df['Action'].str.contains('Short', na=False) & (trade_df['Position Type'] == 'Short')].set_index('Date')
ax1.scatter(short_entries.index, short_entries['Entry Price'], marker='v', color='red', label='Short Entry', alpha=0.7, s=50)

bb_exits = trade_df[trade_df['Position Type'] == 'Close'].set_index('Date')
ax1.scatter(bb_exits.index, bb_exits['Exit Price'], marker='o', color='black', label='BB Exit', alpha=0.5, s=30)

sl_exits = trade_df[trade_df['Position Type'] == 'SL Close'].set_index('Date')
ax1.scatter(sl_exits.index, sl_exits['Exit Price'], marker='D', color='red', label='Stop Loss Exit', alpha=1.0, s=40)

ax1.set_title('Price Action and Bollinger Band Signals')
ax1.set_ylabel('Price ($)')
ax1.legend(loc='upper left')
ax1.grid(True)


ax2 = axes[1]
ax2.plot(equity_df.index, equity_df['Equity'], label='Equity Curve', color='blue')
ax2.plot(equity_df.index, equity_df['Equity'].cummax(), label='All-Time High', color='grey', linestyle='--')
ax2.set_title('Portfolio Equity Curve')
ax2.set_xlabel('Date')
ax2.set_ylabel('Equity ($)')
ax2.legend(loc='upper left')

ax3 = ax2.twinx()
drawdown_pct = ((equity_df['Equity'] / equity_df['Equity'].cummax()) - 1) * 100
ax3.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color='red', alpha=0.3, label='Drawdown (%)')
ax3.set_ylabel('Drawdown (%)', color='red')
ax3.tick_params(axis='y', labelcolor='red')
ax3.axhline(max_drawdown*100, color='red', linestyle=':', label=f'Max Drawdown ({max_drawdown:.2%})')

plt.tight_layout()
plt.show()