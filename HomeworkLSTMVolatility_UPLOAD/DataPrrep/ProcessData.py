# -*- coding: utf-8 -*-.
"""
Created on Sat Nov 20 19:29:09 2021

@author: loren
"""


import pandas as pd

#some SP500 data was missing from CRSP, we will download from Yahoo instead
#df = pd.read_csv("sp500IndexLevelAndReturn_CRSP.csv", parse_dates=True, infer_datetime_format=True)
#df["DATE"]=pd.to_datetime(df["DATE"])
#df.index = df.DATE
#df.drop(["DATE"],axis=1, inplace=True)
#df["abs_sprtrn"] = df.sprtrn.abs()


#some SP500 data was missing, downloading it from yahoo
pd.core.common.is_list_like = pd.api.types.is_list_like #datareader problem probably fixed in next version of datareader
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

start, end = dt.datetime(2016, 1, 28), dt.datetime(2022, 6, 28)
StockList = ['^GSPC']
missing = pd.DataFrame()
for ticker in StockList:
    missing[ticker]  = pdr.get_data_yahoo(ticker, start=start, end=end).loc[:,'Close']
missing.rename(columns={"^GSPC": "spindx"}, inplace=True)
missing["sprtrn"] = (missing.spindx-missing.spindx.shift(1))/missing.spindx.shift(1)
missing["abs_sprtrn"] = missing.sprtrn.abs()
missing = missing.dropna()
#missing.to_csv("missing.csv")
df = missing


df2 = pd.read_csv("VIX_History_CBOE.csv", parse_dates=True, infer_datetime_format=True)
df2 = df2.iloc[:,0:5]
df2["Date"]=pd.to_datetime(df2["DATE"])
df2.index = df2.Date
df2.drop(["Date"],axis=1, inplace=True)
df2.drop(["DATE"],axis=1, inplace=True)
df2.rename(columns={"CLOSE": "vix"}, inplace=True)
df2.rename(columns={"OPEN": "vixo"}, inplace=True)
df2.rename(columns={"HIGH": "vixh"}, inplace=True)
df2.rename(columns={"LOW": "vixl"}, inplace=True)


df3 = pd.read_csv("VIX3M_History_CBOE.csv", parse_dates=True, infer_datetime_format=True)
df3 = df3.iloc[:,0:5]
df3["DATE"]=pd.to_datetime(df3["DATE"])
df3.index = df3.DATE
df3.drop(["DATE", "OPEN", "HIGH","LOW"],axis=1, inplace=True)
df3.rename(columns={"CLOSE": "vix3m"}, inplace=True)

df4 = df3.join(df)
df4 = df4.join(df2)
df5 = df4.fillna(method = "ffill")
df5 = df4.fillna(method = "bfill")
df5["vix3m/vix"] = df5.vix3m/df5.vix
df5.drop(["vix3m"], axis=1, inplace=True)


df5.to_csv("data_without_gex.csv")

#GEX data

df6 = pd.read_csv("data_without_gex.csv", parse_dates=True, infer_datetime_format=True)
df6["DATE"]=pd.to_datetime(df6["DATE"])
df6.index = df6.DATE
df6.drop(["DATE"],axis=1, inplace=True)
df6.drop(["vix3m/vix"],axis=1, inplace=True)

df7 = pd.read_csv("DIX_GEX_SP500.csv", parse_dates=True, infer_datetime_format=True)
df7["date"]=pd.to_datetime(df7["date"])
df7.index = df7.date
#df7.drop(["date", "SP500price","dix"],axis=1, inplace=True)
df7.drop(["date", "price","dix"],axis=1, inplace=True)


df8 = df7.join(df6)

df8.to_csv("data_with_gex.csv")


