import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize(arr):
    tmp = np.array(arr)
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(tmp.reshape(-1, 1))

# Create a dataset class
class Stock:

    def __init__(self, stock_pd_dataframe=pd.read_csv("data/XAUUSD.csv")):
        self.dataframe = stock_pd_dataframe

    def get_mean_price(self):
        df = self.dataframe
        mean_prices = (df["high"] + df["low"] + df["open"] + df["close"]) / 4
        return mean_prices.tolist()

    def get_ma(self, period=10):
        df = self.dataframe
        tmp = df.copy(deep=True)
        tmp.drop(tmp.head(period).index, inplace=True)
        mean_prices = self.get_mean_price()

        prices = []
        for i in range(period):
            prices.append(mean_prices[i:i-period])
            for j in range(len(prices[i])):
                prices[i][j] = prices[i][j] * (period - i)
        
        ma = []
        for j in range(len(prices[0])):
            temp = 0
            for i in range(period):
                temp = temp + prices[i][j]
            temp = temp / np.array(list(range(period + 1))).sum()
            ma.append(temp)
        return ma

    def get_gradient(self, ma):
        gradient = [0, ]

        for i in range(1, len(ma)):
            gradient.append(ma[i] - ma[i-1])
        
        return gradient
    
    def process_data(self):
        processed = self.dataframe.copy(deep=True)
        ma10 = self.get_ma(10)
        grad10 = self.get_gradient(ma10)
        ma50 = self.get_ma(50)
        grad50 = self.get_gradient(ma50)
        ma200 = self.get_ma(200)
        grad200 = self.get_gradient(ma200)
        for i in range(10):
            ma10.insert(0, 0)
            grad10.insert(0, 0)
        for i in range(50):
            ma50.insert(0, 0)
            grad50.insert(0, 0)
        for i in range(200):
            ma200.insert(0, 0)
            grad200.insert(0, 0)
        processed["mean"] = self.get_mean_price()
        processed["ma10"] = ma10
        processed["grad10"] = grad10
        processed["ma50"] = ma50
        processed["grad50"] = grad50
        processed["ma200"] = ma200
        processed["grad200"] = grad200
        processed.drop(processed.head(200).index, inplace = True)

        self.processed = processed
        processed.to_csv("data/Processed.csv", index=False)
        return processed
    
    def get_train_test(self):
        df = self.process_data()
        tmp = pd.DataFrame(df["time"])
        tmp["tick_volume"] = normalize(df["tick_volume"].tolist())
        tmp["mean"] = normalize(df["mean"].tolist())
        tmp["ma10"] = normalize(df["ma10"].tolist())
        tmp["ma50"] = normalize(df["ma50"].tolist())
        tmp["ma200"] = normalize(df["ma200"].tolist())
        tmp["grad10"] = df["grad10"].tolist()
        tmp["grad50"] = df["grad50"].tolist()
        tmp["grad200"] = df["grad200"].tolist()
        tmp.drop(columns=["time"], inplace=True)

        label = []
        for i in range(len(df)):
            try:
                j = 0
                while True:
                    if (df.iloc[i]["open"] - df.iloc[i+j]["close"]) >= 10:
                        label.append(1)
                        break
                    elif (df.iloc[i]["open"] - df.iloc[i+j]["close"]) <= -10:
                        label.append(0)
                        break
                    else:
                        j = j + 1
            except Exception:
                if df.iloc[i]["open"] > df.iloc[-1]["close"]:
                    label.append(0)
                elif df.iloc[i]["open"] < df.iloc[-1]["close"]:
                    label.append(1)
                else:
                    label.append(0)

        tmp["label"] = label
        tmp.drop(tmp.tail(10).index, inplace = True)

        tmpx = []
        tmpy = []
        for i in range(len(tmp)):
            if i >= 20:
                temp = []
                for j in range(20):
                    temp.append(tmp.iloc[i-20+j, 0:-1].tolist())
                tmpx.append(temp)
                tmpy.append(tmp.iloc[i, -1])
            else:
                i = i + 1
        
        features = []
        labels = []
        for i in range(len(tmpy)):
            if tmpx[i][19][5] * tmpx[i][18][5] < 0:
                features.append(tmpx[i])
                labels.append(tmpy[i])
        
        features = np.array(features)
        labels = np.array(labels)

        split_point = int(len(labels) * 0.9)
        X_train, X_test = features[0:split_point], features[split_point:]
        y_train, y_test = labels[0:split_point], labels[split_point:]

        np.save("data/X_train", X_train)
        np.save("data/X_test", X_test)
        np.save("data/y_train", y_train)
        np.save("data/y_test", y_test)

if __name__ == "__main__":
    
    xau = Stock()
    xau.get_train_test()