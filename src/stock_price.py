from pykrx import stock

def get_stock_price(start_date, end_date, stock_code):
    df = stock.get_market_ohlcv(start_date, end_date, stock_code)
    print(df.head(3))
    return df