from src.sentiment_analysis_combine import get_naver_finance_board
from src.stock_price import get_stock_price
import pandas as pd

result = get_stock_price()
print(result)

result2 = get_naver_finance_board("041510", 30, 2023, 2, 1)

result2['날짜'] = pd.to_datetime(result2['날짜'])
merged_df = pd.merge(result, result2, on='날짜')
print(merged_df)