import requests
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time

class NewsCollector:
    def __init__(self):
        self.base_url = "https://www.coindesk.com"
        
    def get_crypto_news(self, limit=50):
        """코인데스크에서 암호화폐 뉴스 가져오기"""
        try:
            news_data = []
            page = 1
            
            while len(news_data) < limit:
                url = f"{self.base_url}/page/{page}/"
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                articles = soup.find_all('article')
                for article in articles:
                    if len(news_data) >= limit:
                        break
                        
                    headline = article.find('h3').text.strip()
                    content = article.find('p').text.strip()
                    date = article.find('time')['datetime']
                    
                    news_data.append({
                        'date': pd.to_datetime(date),
                        'headline': headline,
                        'content': content
                    })
                
                page += 1
                time.sleep(1)  # 웹사이트 부하 방지
                
            return pd.DataFrame(news_data)
            
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return None 