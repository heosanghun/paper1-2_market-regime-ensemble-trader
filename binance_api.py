from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta

class BinanceAPI:
    def __init__(self, api_key=None, api_secret=None):
        self.client = Client(api_key, api_secret)
        
    def get_candlestick_data(self, symbol='BTCUSDT', interval='1h', limit=100):
        """바이낸스에서 캔들스틱 데이터 가져오기"""
        try:
            # 캔들스틱 데이터 요청
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 데이터 타입 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching candlestick data: {str(e)}")
            return None 