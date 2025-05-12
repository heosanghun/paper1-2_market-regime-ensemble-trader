import matplotlib
matplotlib.use('Agg')
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import os
from generate_charts import add_indicators, save_chart, is_valid_chart

# 저장 경로
save_dir = r"D:/drl-candlesticks-trader-main1/captures/1d"
os.makedirs(save_dir, exist_ok=True)

client = Client()

# 5년치 데이터 범위
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)
window_days = 180  # 6개월 단위로 변경
cur = start_date

while cur + timedelta(days=window_days) <= end_date:
    s = cur.strftime("%Y-%m-%d")
    e = (cur + timedelta(days=window_days)).strftime("%Y-%m-%d")
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, s, e)
    df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qv','trades','tb','tq','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    if len(df) < 5:
        print(f"{s}~{e} 차트 저장 스킵: 데이터 부족")
        cur += timedelta(days=1)
        continue
    df = add_indicators(df)
    valid, reason = is_valid_chart(df)
    if valid:
        save_chart(df, "1d", s, e, save_dir)
    else:
        print(f"{s}~{e} 차트 저장 스킵: {reason}")
    cur += timedelta(days=window_days) 