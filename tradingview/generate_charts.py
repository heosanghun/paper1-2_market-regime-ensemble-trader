import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import mplfinance as mpf
from binance.client import Client
from datetime import datetime, timedelta
import ta
import os

def get_binance_ohlcv(symbol, interval, start, end):
    client = Client()
    klines = client.get_historical_klines(symbol, interval, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    df = pd.DataFrame(klines, columns=['timestamp','open','high','low','close','volume','close_time','qv','trades','tb','tq','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df

def add_indicators(df):
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    return df

def is_valid_chart(df):
    # 캔들 개수 5개 이상, 모든 주요 지표 컬럼에 NaN이 아닌 값이 1개 이상
    if len(df) < 5:
        return False, "캔들 개수 부족"
    for col in ['bb_upper', 'bb_lower', 'RSI', 'MACD', 'MACD_signal']:
        if col not in df or df[col].dropna().empty:
            return False, f"{col} NaN만 존재"
    return True, ""

def save_chart(df, interval, start_date, end_date, save_dir):
    apds = [
        mpf.make_addplot(df['bb_upper'], color='blue'),
        mpf.make_addplot(df['bb_lower'], color='blue'),
        mpf.make_addplot(df['RSI'], panel=2, color='purple', title='RSI'),
        mpf.make_addplot(df['MACD'], panel=3, color='blue', title='MACD'),
        mpf.make_addplot(df['MACD_signal'], panel=3, color='orange')
    ]
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        rc={'axes.facecolor':'black', 'figure.facecolor':'black'},
        marketcolors=mpf.make_marketcolors(up='lime', down='red', edge='inherit', wick='white', volume='in')
    )
    filename = os.path.join(save_dir, f"BTCUSDT_{interval}_{start_date}_{end_date}_binance.png")
    mpf.plot(
        df,
        type='candle',
        style=style,
        addplot=apds,
        volume=True,
        figsize=(8,8),
        savefig=filename,
        tight_layout=True
    )
    print(f"{interval} {start_date}~{end_date} 차트 저장 완료!")

def save_all_windows(symbol, interval, window_days, save_path, start_date, end_date, min_candles=5):
    cur = start_date
    window = timedelta(days=window_days)
    while cur + window <= end_date:
        s = cur.strftime("%Y-%m-%d")
        e = (cur + window).strftime("%Y-%m-%d")
        df = get_binance_ohlcv(symbol, interval, cur, cur + window)
        df = add_indicators(df)
        valid, reason = is_valid_chart(df)
        if valid:
            try:
                save_chart(df, interval, s, e, save_path)
            except Exception as ex:
                print(f"{interval} {s}~{e} 차트 저장 실패: {ex}")
            cur += window
        else:
            print(f"{interval} {s}~{e} 차트 저장 스킵: {reason}")
            cur += timedelta(days=1)

def save_event_periods(symbol, interval, save_path, event_periods, min_candles=5):
    for ev_start, ev_end in event_periods:
        s = ev_start
        e = ev_end
        df = get_binance_ohlcv(symbol, interval, datetime.strptime(s, "%Y-%m-%d"), datetime.strptime(e, "%Y-%m-%d"))
        df = add_indicators(df)
        valid, reason = is_valid_chart(df)
        if valid:
            try:
                save_chart(df, interval, s, e, save_path)
            except Exception as ex:
                print(f"{interval} {s}~{e} 이벤트 차트 저장 실패: {ex}")
        else:
            print(f"{interval} {s}~{e} 이벤트 차트 저장 스킵: {reason}")

if __name__ == "__main__":
    symbol = "BTCUSDT"
    intervals = {
        "1d": (Client.KLINE_INTERVAL_1DAY, 30),      # 30일
        "1w": (Client.KLINE_INTERVAL_1WEEK, 180)     # 180일
    }
    base_dir = "./captures"
    os.makedirs(base_dir, exist_ok=True)
    # 5년치 전체 구간 생성
    start_date = datetime.now() - timedelta(days=365*5)
    end_date = datetime.now()
    # 주요 이벤트 구간
    event_periods = [
        ('2022-11-01', '2022-11-30'), # FTX
        ('2022-05-01', '2022-05-31'), # 테라/루나
        ('2020-03-01', '2020-03-31'), # 코로나
        ('2020-05-01', '2020-05-31'), # 반감기
    ]
    for name, (interval, window_days) in intervals.items():
        save_path = os.path.join(base_dir, name)
        os.makedirs(save_path, exist_ok=True)
        save_all_windows(symbol, interval, window_days, save_path, start_date, end_date)
        save_event_periods(symbol, interval, save_path, event_periods)
    # 최근 30일 구간 샘플 저장
    save_path = os.path.join(base_dir, "1d_sample")
    os.makedirs(save_path, exist_ok=True)
    # 최근 30일 구간 샘플 저장
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    df = get_binance_ohlcv(symbol, Client.KLINE_INTERVAL_1DAY, start_date, end_date)
    df = add_indicators(df)
    valid, reason = is_valid_chart(df)
    if valid:
        save_chart(df, "1d", start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), save_path)
    else:
        print(f"1d 샘플 차트 저장 스킵: {reason}") 