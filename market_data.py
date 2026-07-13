import pandas as pd
import yfinance as yf


def load_market(ticker="SPY", start="2010-01-01", end="2024-01-01", horizon=1):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})

    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["vol10"] = df["ret1"].rolling(10).std()
    df["ma_ratio"] = df["close"] / df["close"].rolling(20).mean()
    df["mom10"] = df["close"] / df["close"].shift(10) - 1
    df["vol_chg"] = df["volume"].pct_change()
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)  # direction horizon days out

    df = df.dropna()
    features = ["ret1", "ret5", "vol10", "ma_ratio", "mom10", "vol_chg"]
    return df[features].reset_index(drop=True), df["target"].reset_index(drop=True)