import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5

# connect to MetaTrader 5
if not mt5.initialize():
    print("Failed to initialize MetaTrader 5")
    exit()

# specify the symbol and timeframe to retrieve data for
symbol = "XAUUSD"
timeframe = mt5.TIMEFRAME_M30
d = datetime.now()
dt = datetime(d.year, d.month, d.day)

# retrieve the historical data
rates = mt5.copy_rates_from(symbol, timeframe, dt, 10000)

# convert the data into a pandas dataframe
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")

# save the data to a CSV file
df.to_csv("data/XAUUSD.csv", index=False)

# disconnect from MetaTrader 5
mt5.shutdown()