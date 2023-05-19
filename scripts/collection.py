import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5


def get_data(records, timeframe="H4", symbol = "XAUUSD", name="gold"):

    # connect to MetaTrader 5
    if not mt5.initialize():
        print("Failed to initialize MetaTrader 5")
        return()

    # specify the symbol and timeframe to retrieve data for
    if timeframe == "M5":
        timeframe1 = mt5.TIMEFRAME_M5
    elif timeframe == "M15":
        timeframe1 = mt5.TIMEFRAME_M15
    elif timeframe == "M30":
        timeframe1 = mt5.TIMEFRAME_M30
    elif timeframe == "H1":
        timeframe1 = mt5.TIMEFRAME_H1
    elif timeframe == "H4":
        timeframe1 = mt5.TIMEFRAME_H4
    elif timeframe == "H12":
        timeframe1 = mt5.TIMEFRAME_H12
    elif timeframe == "D1":
        timeframe1 = mt5.TIMEFRAME_D1
    elif timeframe == "W1":
        timeframe1 = mt5.TIMEFRAME_W1
    elif timeframe == "M1":
        timeframe1 = mt5.TIMEFRAME_M1
    else:
        timeframe1 = mt5.TIMEFRAME_H1
    d = datetime.now()
    dt = datetime(d.year, d.month, d.day)

    # retrieve the historical data
    rates = mt5.copy_rates_from(symbol, timeframe1, dt, records)

    # convert the data into a pandas dataframe
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # save the data to a CSV file
    df.to_csv(f"data/{name}_{timeframe}.csv", index=False)

    # disconnect from MetaTrader 5
    mt5.shutdown()