import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import argparse
import time

from regime_adaptive_trading_system import RegimeAdaptiveTradingSystem

def download_stock_data(symbol, period='1y', interval='1h'):
    """
    야후 파이낸스에서 주식 데이터를 다운로드합니다.
    """
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {symbol} 데이터 다운로드 시작...")
    data = yf.download(symbol, period=period, interval=interval)
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 다운로드 완료: {len(data)} 개의 데이터 포인트 (소요시간: {elapsed:.2f}초)")
    return data

def prepare_timeframes(data_1h):
    """
    여러 타임프레임 데이터를 준비합니다.
    """
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 다중 타임프레임 데이터 준비 시작...")
    
    # 1시간 데이터 복사
    data_1h_copy = data_1h.copy()
    
    # 4시간 데이터 생성
    data_4h = data_1h_copy.resample('4H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # 일별 데이터 생성
    data_1d = data_1h_copy.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    # 컬럼명 소문자로 변환
    data_1h_copy.columns = [col.lower() for col in data_1h_copy.columns]
    data_4h.columns = [col.lower() for col in data_4h.columns]
    data_1d.columns = [col.lower() for col in data_1d.columns]
    
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 다중 타임프레임 데이터 준비 완료 (소요시간: {elapsed:.2f}초)")
    
    return {
        '1h': data_1h_copy,
        '4h': data_4h,
        '1d': data_1d
    }

def run_backtest(system, test_data, initial_capital=10000):
    """
    시장 레짐 적응형 트레이딩 시스템의 백테스트를 실행합니다.
    """
    total_bars = len(test_data)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 백테스트 시작: {total_bars}개 막대")
    start_time = time.time()
    last_update_time = start_time
    
    # 진행 상황 추적
    results = []
    equity_curve = [initial_capital]
    current_capital = initial_capital
    
    # 각 bar마다 시스템 실행
    for i in range(1, len(test_data)):
        # 새 bar 데이터
        new_bar = test_data.iloc[[i]]
        
        # 시스템 업데이트 및 결정 가져오기
        decision = system.process_bar(new_bar)
        
        # 결과 저장
        if 'pnl_amount' in decision:
            current_capital += decision['pnl_amount']
            
        equity_curve.append(current_capital)
        results.append(decision)
        
        # 진행 상황 출력 (10% 단위로 업데이트)
        current_time = time.time()
        if i % max(1, total_bars // 20) == 0 or i == len(test_data) - 1 or current_time - last_update_time > 30:
            elapsed = current_time - start_time
            progress = i / total_bars
            estimated_total = elapsed / progress if progress > 0 else 0
            remaining = estimated_total - elapsed
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 진행: {i}/{total_bars} ({progress*100:.1f}%)")
            print(f"  경과 시간: {timedelta(seconds=int(elapsed))}, 예상 남은 시간: {timedelta(seconds=int(remaining))}")
            print(f"  현재 자본금: ${current_capital:.2f}, 변화: {(current_capital/initial_capital-1)*100:.2f}%")
            
            last_update_time = current_time
    
    total_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 백테스트 완료! 총 소요시간: {timedelta(seconds=int(total_time))}")
    
    # 성과 지표 계산
    performance = system.get_performance_metrics()
    
    # 결과를 데이터프레임으로 변환
    equity_df = pd.DataFrame({
        'timestamp': test_data.index[1:],
        'equity': equity_curve[1:]
    })
    
    return equity_df, performance, results

def plot_results(equity_df, symbol, timeframe_data, system):
    """
    백테스트 결과를 시각화합니다.
    """
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 결과 시각화 시작...")
    
    # 결과 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    
    # 1. 자본금 곡선
    plt.figure(figsize=(14, 7))
    plt.plot(equity_df['timestamp'], equity_df['equity'])
    plt.title(f'Equity Curve - {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True)
    plt.savefig("results/equity_curve.png")
    
    # 2. 가격 차트와 거래 표시
    price_data = timeframe_data['1h']
    trades = pd.DataFrame(system.trades_history)
    
    if not trades.empty and 'timestamp' in trades.columns:
        plt.figure(figsize=(14, 7))
        plt.plot(price_data.index, price_data['close'], label='Price')
        
        # 매수/매도 지점 표시
        buy_trades = trades[trades['action'] == 'BUY']
        sell_trades = trades[trades['action'] == 'SELL']
        close_trades = trades[trades['action'] == 'CLOSE']
        
        if not buy_trades.empty:
            plt.scatter(
                buy_trades['timestamp'],
                buy_trades['price'],
                marker='^',
                color='green',
                s=100,
                label='Buy'
            )
            
        if not sell_trades.empty:
            plt.scatter(
                sell_trades['timestamp'],
                sell_trades['price'],
                marker='v',
                color='red',
                s=100,
                label='Sell'
            )
            
        if not close_trades.empty:
            plt.scatter(
                close_trades['timestamp'],
                close_trades['price'],
                marker='x',
                color='blue',
                s=100,
                label='Close'
            )
        
        plt.title(f'Price Chart with Trades - {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig("results/price_chart_with_trades.png")
    
    # 3. 레짐 분포
    regime_history = pd.DataFrame(system.regime_history)
    if not regime_history.empty:
        regimes = regime_history['regime'].value_counts()
        
        plt.figure(figsize=(10, 6))
        regimes.plot(kind='bar')
        plt.title('Market Regime Distribution')
        plt.xlabel('Regime')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/regime_distribution.png")
    
    # 4. 월별 수익률 히트맵
    if not equity_df.empty:
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        equity_df['date'] = pd.to_datetime(equity_df['timestamp'])
        equity_df['year'] = equity_df['date'].dt.year
        equity_df['month'] = equity_df['date'].dt.month
        
        monthly_returns = equity_df.groupby(['year', 'month'])['daily_return'].sum().unstack()
        
        plt.figure(figsize=(12, 8))
        plt.imshow(monthly_returns, cmap='RdYlGn')
        plt.colorbar(label='Monthly Return')
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.yticks(range(len(monthly_returns.index)), monthly_returns.index)
        plt.xticks(range(len(monthly_returns.columns)), monthly_returns.columns)
        
        for i in range(len(monthly_returns.index)):
            for j in range(len(monthly_returns.columns)):
                try:
                    value = monthly_returns.iloc[i, j]
                    plt.text(j, i, f'{value:.2%}', ha='center', va='center', 
                            color='white' if abs(value) > 0.05 else 'black')
                except:
                    pass
        
        plt.savefig("results/monthly_returns_heatmap.png")
    
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 결과 시각화 완료 (소요시간: {elapsed:.2f}초)")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 결과가 'results/' 디렉토리에 저장되었습니다.")

def print_performance_summary(performance):
    """
    성과 지표 요약을 출력합니다.
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ===== 성과 요약 =====")
    print(f"총 수익률: {performance['total_return']:.2%}")
    print(f"최대 낙폭: {performance['drawdown']:.2%}")
    print(f"총 거래 수: {performance['total_trades']}")
    print(f"승률: {performance['win_rate']:.2%}")
    print(f"수익:손실 비율: {performance['win_loss_ratio']:.2f}")
    print(f"최대 연속 손실: {performance['max_consecutive_losses']}")
    print(f"현재 리스크 레벨: {performance['current_risk_level']}")
    
    print("\n시장 레짐 분포:")
    for regime, count in performance['regime_distribution'].items():
        print(f"  {regime}: {count}")

def save_results_to_csv(equity_df, trades_history, performance, symbol):
    """
    백테스트 결과를 CSV 파일로 저장합니다.
    """
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 결과 CSV 저장 중...")
    
    # 결과 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    
    # 자본금 곡선 저장
    equity_df.to_csv(f"results/equity_curve_{symbol}.csv", index=False)
    
    # 거래 내역 저장
    trades_df = pd.DataFrame(trades_history)
    if not trades_df.empty:
        trades_df.to_csv(f"results/trades_{symbol}.csv", index=False)
    
    # 성과 지표 저장
    performance_df = pd.DataFrame([performance])
    performance_df.to_csv(f"results/performance_{symbol}.csv", index=False)
    
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 결과 CSV 저장 완료 (소요시간: {elapsed:.2f}초)")

def main():
    parser = argparse.ArgumentParser(description='시장 레짐 적응형 트레이딩 시스템')
    parser.add_argument('--symbol', type=str, default='SPY', help='거래할 종목의 티커 심볼')
    parser.add_argument('--period', type=str, default='1y', help='다운로드할 데이터 기간 (예: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
    parser.add_argument('--interval', type=str, default='1h', help='데이터 간격 (예: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)')
    parser.add_argument('--initial_capital', type=float, default=10000, help='초기 자본금')
    parser.add_argument('--max_position_size', type=float, default=0.2, help='최대 포지션 크기 (자본의 %)')
    args = parser.parse_args()
    
    overall_start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 시장 레짐 적응형 트레이딩 시스템 실행 시작")
    print(f"설정: 종목={args.symbol}, 기간={args.period}, 간격={args.interval}, 초기자본=${args.initial_capital}")
    
    # 1. 데이터 다운로드
    data = download_stock_data(args.symbol, args.period, args.interval)
    
    # 2. 다중 타임프레임 데이터 준비
    timeframe_data = prepare_timeframes(data)
    
    # 3. 시스템 초기화
    system_start = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 시스템 초기화 시작...")
    
    system = RegimeAdaptiveTradingSystem(
        initial_capital=args.initial_capital,
        max_position_size=args.max_position_size
    )
    
    # 4. 데이터 로드 (처음 절반은 학습용)
    train_size = len(timeframe_data['1h']) // 2
    
    train_data = {
        '1h': timeframe_data['1h'].iloc[:train_size],
        '4h': timeframe_data['4h'],
        '1d': timeframe_data['1d']
    }
    
    test_data = timeframe_data['1h'].iloc[train_size-100:]  # 100개 데이터 오버랩
    
    # 5. 시스템에 학습 데이터 로드
    system.load_data(train_data)
    
    system_elapsed = time.time() - system_start
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 시스템 초기화 완료, 학습 데이터: {train_size} 개 막대 (소요시간: {system_elapsed:.2f}초)")
    
    # 6. 백테스트 실행
    equity_df, performance, results = run_backtest(
        system, test_data, args.initial_capital
    )
    
    # 7. 결과 시각화
    plot_results(equity_df, args.symbol, timeframe_data, system)
    
    # 8. 성과 요약 출력
    print_performance_summary(performance)
    
    # 9. 결과 저장
    save_results_to_csv(equity_df, system.trades_history, performance, args.symbol)
    
    overall_elapsed = time.time() - overall_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 총 실행 시간: {timedelta(seconds=int(overall_elapsed))}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main() 