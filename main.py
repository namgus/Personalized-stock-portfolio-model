from src.sentiment_analysis_average import get_naver_finance_board
from src.stock_price import get_stock_price
import pandas as pd

result = get_stock_price("20220601", "20221231", "005930")
print(result)

result2 = get_naver_finance_board("005930", 500000, 2022, 6, 1)

result2['날짜'] = pd.to_datetime(result2['날짜'])
merged_df = pd.merge(result, result2, on='날짜')

merged_df.to_csv('final_main_average_data.csv', index=False)

print(merged_df)