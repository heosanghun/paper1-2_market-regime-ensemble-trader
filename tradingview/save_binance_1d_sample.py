import matplotlib
matplotlib.use('Agg')
import mplfinance as mpf
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

# 바이낸스 클라이언트 생성
client = Client()

# 최근 30일 데이터 가져오기
end = datetime.now()
start = end - timedelta(days=30)
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qv','trades','tb','tq','ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
for col in ['open','high','low','close','volume']:
    df[col] = df[col].astype(float)

# 차트 이미지 저장
mpf.plot(df, type='candle', volume=True, style='charles', savefig='binance_1d_sample.png')
print('binance_1d_sample.png 저장 완료!') 