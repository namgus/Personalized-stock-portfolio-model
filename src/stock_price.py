from pykrx import stock

def get_stock_price():
    df = stock.get_market_ohlcv("20220720", "20220810", "005930")
    print(df.head(3))
    return df