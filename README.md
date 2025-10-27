# Quant_Global
Problem_working
A simple GARCH-driven Bollinger-Band Mean reversion strategy

The code is based on a GARCH(1,1) model, used to generate a semi-realistic time-series of an instrument. Initially started with a simple random walk equation, moved to geometric brownian motion, but finally fixated on the GARCH model its closer to a realistic market data. 

Bollinger bands are drawn to capture the price movement within the SMA of any given lookback period, the pnl was drawn with the logic being, price<lower_band, we buy and price>upper_band we sell and also we simultaneously buy/short the stock, so as to always maintain an open position. 

the plots done, show the metrics calculated, the max_dd, the stock_price with the bands, entry and exit points.

A very straightforward implementation of a standard strategy in finance.
