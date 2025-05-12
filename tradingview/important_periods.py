import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import ta

def get_binance_data(symbol, interval, start_date, end_date):
    client = Client()
    klines = client.get_historical_klines(
        symbol,
        interval,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignored'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def identify_important_periods(df, window=14):
    # RSI 계산
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi()
    
    # 볼린저 밴드
    bb = ta.volatility.BollingerBands(df['close'], window=window)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # 일일 변동성
    df['daily_change'] = abs(df['close'].pct_change()) * 100
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # 중요 구간 식별
    important_periods = []
    
    # 1. RSI 과매수/과매도
    rsi_extreme = df[((df['rsi'] > 70) | (df['rsi'] < 30))]
    
    # 2. 볼린저 밴드 이탈
    bb_break = df[((df['close'] > df['bb_high']) | (df['close'] < df['bb_low']))]
    
    # 3. MACD 크로스
    df['macd_cross'] = np.where(
        ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))) |
        ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))),
        True, False
    )
    macd_crosses = df[df['macd_cross']]
    
    # 4. 높은 변동성
    volatility_threshold = df['daily_change'].mean() * 2
    high_volatility = df[df['daily_change'] > volatility_threshold]
    
    # 5. 주요 이벤트 기간
    events = {
        'FTX_COLLAPSE': ('2022-11-01', '2022-11-30'),
        'TERRA_LUNA': ('2022-05-01', '2022-05-31'),
        'COVID_CRASH': ('2020-03-01', '2020-03-31'),
        'BTC_HALVING_2020': ('2020-05-01', '2020-05-31'),
    }
    
    # 결과 통합 및 중복 제거
    important_dates = set()
    
    for idx in rsi_extreme.index:
        important_dates.add(df.loc[idx, 'timestamp'].strftime('%Y-%m-%d'))
    
    for idx in bb_break.index:
        important_dates.add(df.loc[idx, 'timestamp'].strftime('%Y-%m-%d'))
    
    for idx in macd_crosses.index:
        important_dates.add(df.loc[idx, 'timestamp'].strftime('%Y-%m-%d'))
    
    for idx in high_volatility.index:
        important_dates.add(df.loc[idx, 'timestamp'].strftime('%Y-%m-%d'))
    
    for event_period in events.values():
        start, end = event_period
        current = datetime.strptime(start, '%Y-%m-%d')
        end = datetime.strptime(end, '%Y-%m-%d')
        while current <= end:
            important_dates.add(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
    
    return sorted(list(important_dates))

def get_important_periods(symbol="BTCUSDT", interval="1d", start_date=None, end_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=365)
    if end_date is None:
        end_date = datetime.now()
    
    # 데이터 수집
    df = get_binance_data(symbol, interval, start_date, end_date)
    
    # 중요 구간 식별
    important_dates = identify_important_periods(df)
    
    # 결과 출력
    print(f"총 {len(important_dates)}개의 중요 구간이 식별되었습니다.")
    print("주요 날짜:")
    for date in important_dates[:10]:  # 처음 10개만 출력
        print(f"- {date}")
    
    return important_dates

if __name__ == "__main__":
    # 최근 1년 데이터에서 중요 구간 식별
    important_dates = get_important_periods()
    
    # 결과를 파일로 저장
    with open("important_dates.txt", "w") as f:
        f.write("\n".join(important_dates)) 