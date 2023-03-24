from datetime import datetime

import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# pre-trained model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")

labels = ['Negative', 'Neutral', 'Positive']

sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

def get_sentiment_analysis(text):
    # Classify the sentiment of the input sentence
    if len(text) > 512:
        text = text[:512] # restrict model input 512
    result = sentiment_analysis(text)

    max_score_label = max(result[0], key=lambda x: x['score'])['label']

    score_dict = {'negative': -1, 'neutral': 0, 'positive': 1}
    score = sum([score_dict[r['label']] * r['score'] for r in result[0]])

    # Return highest confidence score label
    return score


def get_naver_finance_board(codes, max_page, year, month, day):
    start = time.time()

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36'
    }

    time_pass_flag = 0
    result = []

    target_date = datetime(year, month, day).date()

    for page_num in range(4000, max_page + 1):
        url = f"https://finance.naver.com/item/board.naver?code={codes}&page={page_num}"
        try:
            response = requests.get(url, headers=headers)
            html = response.content
            soup = BeautifulSoup(html.decode('euc-kr', 'replace'), 'html.parser')
            table = soup.find('table', {'class': 'type2'})
            rows = table.select('tbody > tr')
            for row in rows[2:]:
                cols = row.select('td')
                if len(cols) >= 6:
                    date = cols[0].select_one('span').get_text(strip=True)
                    datetime_object = datetime.strptime(date, "%Y.%m.%d %H:%M")
                    if datetime_object.date() < target_date:
                        time_pass_flag = 1
                        break
                    title = cols[1].select_one('a').get('title').strip()
                    views = cols[3].select_one('span').get_text(strip=True)
                    upvote = cols[4].select_one('strong').get_text(strip=True)
                    downvote = cols[5].select_one('strong').get_text(strip=True)
                    result.append([date, title, views, upvote, downvote])

            if time_pass_flag:
                break
        
            # Check crawling process
            if page_num % 100 == 0:
                now = time.time()
                print(page_num, now - start)

        except requests.exceptions.RequestException as e:
            print(f"Error occurred while fetching page {page_num}: {e}")
            break

    df = pd.DataFrame(result, columns=['날짜', '제목', '조회', '공감', '비공감'])

    # combine titles by same date
    df['날짜'] = pd.to_datetime(df['날짜'], format='%Y.%m.%d %H:%M').dt.date

    df.drop(columns = ['조회', '공감', '비공감'], inplace=True)

    df['감성분석'] = df['제목'].apply(get_sentiment_analysis)

    df = df.groupby('날짜')['감성분석'].mean().reset_index()

    print(df)

    return df