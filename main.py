from src.sentiment_analysis_combine import get_naver_finance_board
from src.stock_price import get_stock_price

result = get_stock_price()
print(result)

get_naver_finance_board("041510", 500, 2023, 2, 1)