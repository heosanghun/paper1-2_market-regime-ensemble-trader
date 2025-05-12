#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
백테스팅 모듈
- 트레이딩 전략을 과거 데이터에 대해 검증
- 수익률, 최대 낙폭, 승률 등 계산
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Backtester')

class Backtester:
    """백테스팅 클래스"""
    
    def __init__(self, config):
        """초기화"""
        self.config = config
        
        # 트레이딩 설정
        trading_config = config.get('trading', {})
        
        # 트레이딩 설정값
        self.initial_balance = trading_config.get('initial_balance', 10000)  # USD
        self.trade_amount = trading_config.get('trade_amount', 100)  # USD
        self.fee_rate = trading_config.get('fee_rate', 0.001)  # 0.1%
        
        # 백테스팅 설정
        self.backtest_days = trading_config.get('backtest_days', 20)
        
        logger.info("백테스터 초기화 완료")
    
    def run(self, price_data, signals):
        """백테스팅 실행"""
        logger.info(f"백테스팅 실행 중... (초기 자본금: ${self.initial_balance})")
        
        # 가격 데이터와 시그널 병합
        if isinstance(price_data, pd.DataFrame):
            if 'close' in price_data.columns:
                # 이미 OHLCV 데이터인 경우
                df = price_data.copy()
            else:
                # 종가 데이터만 있는 경우
                df = pd.DataFrame({
                    'close': price_data,
                    'open': price_data,
                    'high': price_data,
                    'low': price_data,
                    'volume': 0
                })
        else:
            # 가격 데이터가 시리즈나 배열인 경우
            df = pd.DataFrame({
                'close': price_data,
                'open': price_data,
                'high': price_data,
                'low': price_data,
                'volume': 0
            })
        
        # 시그널 병합
        if not signals.empty:
            common_idx = df.index.intersection(signals.index)
            if len(common_idx) > 0:
                df = df.loc[common_idx]
                signals = signals.loc[common_idx]
                df['signal'] = signals['signal']
            else:
                # 인덱스가 일치하지 않는 경우
                logger.warning("가격 데이터와 시그널의 인덱스가 일치하지 않습니다. 가상 시그널을 생성합니다.")
                df['signal'] = np.random.choice(['buy', 'sell', 'hold'], size=len(df), p=[0.3, 0.3, 0.4])
        else:
            # 시그널이 없는 경우
            logger.warning("시그널이 없습니다. 가상 시그널을 생성합니다.")
            df['signal'] = np.random.choice(['buy', 'sell', 'hold'], size=len(df), p=[0.3, 0.3, 0.4])
        
        # 백테스팅 결과
        balance = self.initial_balance
        position = 0  # 포지션 수량
        entry_price = 0  # 진입 가격
        trades = []  # 거래 이력
        
        # 자본금 추적
        equity_curve = [balance]
        dates = [df.index[0] - timedelta(days=1)]  # 초기 날짜
        
        # 백테스팅 실행
        for i, (idx, row) in enumerate(df.iterrows()):
            # 현재 가격
            current_price = row['close']
            
            # 현재 가치 계산
            current_value = balance + position * current_price
            equity_curve.append(current_value)
            dates.append(idx)
            
            # 시그널에 따른 거래 실행
            if row['signal'] == 'buy' and position == 0:
                # 매수
                qty = self.trade_amount / current_price
                fee = self.trade_amount * self.fee_rate
                
                if balance >= self.trade_amount + fee:
                    balance -= self.trade_amount + fee
                    position = qty
                    entry_price = current_price
                    
                    # 거래 기록
                    trades.append({
                        'timestamp': idx,
                        'action': 'buy',
                        'price': current_price,
                        'quantity': qty,
                        'fee': fee,
                        'balance': balance
                    })
            
            elif row['signal'] == 'sell' and position > 0:
                # 매도
                value = position * current_price
                fee = value * self.fee_rate
                balance += value - fee
                
                # 거래 기록
                trades.append({
                    'timestamp': idx,
                    'action': 'sell',
                    'price': current_price,
                    'quantity': position,
                    'fee': fee,
                    'balance': balance,
                    'profit': (current_price - entry_price) * position - fee
                })
                
                # 포지션 청산
                position = 0
                entry_price = 0
        
        # 마지막 포지션 청산
        if position > 0:
            # 매도
            current_price = df['close'].iloc[-1]
            value = position * current_price
            fee = value * self.fee_rate
            balance += value - fee
            
            # 거래 기록
            trades.append({
                'timestamp': df.index[-1],
                'action': 'sell',
                'price': current_price,
                'quantity': position,
                'fee': fee,
                'balance': balance,
                'profit': (current_price - entry_price) * position - fee
            })
        
        # 결과 계산
        equity_series = pd.Series(equity_curve, index=dates)
        
        # 승률 계산
        profit_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
        total_trades = sum(1 for t in trades if t.get('action') == 'sell')
        win_rate = profit_trades / total_trades if total_trades > 0 else 0
        
        # 최대 낙폭 계산
        max_drawdown = 0
        peak = equity_curve[0]
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # 총 수익률
        total_return = (balance - self.initial_balance) / self.initial_balance
        
        # 결과 저장
        results = {
            'equity_curve': equity_series,
            'trades': trades,
            'metrics': {
                'initial_balance': self.initial_balance,
                'final_balance': balance,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'total_trades': total_trades,
                'win_trades': profit_trades,
                'loss_trades': total_trades - profit_trades,
                'win_rate': win_rate,
                'win_rate_pct': win_rate * 100
            }
        }
        
        logger.info(f"백테스팅 완료: 수익률 {total_return*100:.2f}%, 최대 낙폭 {max_drawdown*100:.2f}%, 승률 {win_rate*100:.2f}%")
        
        return results 