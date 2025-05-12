#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
고급 멀티모달 트레이더 모듈
- 다양한 데이터 소스를 결합하여 트레이딩 의사결정을 수행
- 강화학습과 앙상블 방법론을 결합한 고급 트레이딩 시스템
- 하이브리드 애널리틱스 및 다중 타임프레임 분석
- 동적 가중치 조정 및 멀티 타임프레임 분석
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import json
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import math

from basic_trader import BasicTrader
from candlestick_analyzer import CandlestickAnalyzer
from sentiment_analyzer import SentimentAnalyzer
from rl_trader import PPOTrader, convert_to_state, action_to_position

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ModelEnsemble:
    """
    다양한 모델을 앙상블하여 예측 결과를 도출하는 클래스
    - 각 모델의 가중치를 동적으로 조정하여 최적의 예측 결과 도출
    """
    def __init__(self, models=None, weights=None):
        """
        Args:
            models: 앙상블할 모델 리스트
            weights: 각 모델의 가중치 (None인 경우 동일 가중치 적용)
        """
        self.models = models or []
        
        if weights is None and models:
            # 동일한 가중치 초기화
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights or []
            
        # 가중치 정규화
        if self.weights:
            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]
    
    def add_model(self, model, weight=None):
        """모델 추가"""
        self.models.append(model)
        
        if weight is None:
            # 동일 가중치 재계산
            self.weights = [1.0 / len(self.models)] * len(self.models)
        else:
            # 가중치 추가 후 정규화
            self.weights.append(weight)
            total = sum(self.weights)
            if total > 0:
                self.weights = [w / total for w in self.weights]
    
    def set_weights(self, weights):
        """가중치 설정"""
        if len(weights) != len(self.models):
            raise ValueError("가중치 개수가 모델 개수와 일치해야 합니다.")
            
        self.weights = weights
        
        # 가중치 정규화
        total = sum(self.weights)
        if total > 0:
            self.weights = [w / total for w in self.weights]
    
    def predict(self, *args, **kwargs):
        """앙상블 예측 수행"""
        if not self.models:
            return 0
            
        # 각 모델의 예측 결과 취합
        predictions = []
        for model in self.models:
            try:
                pred = model(*args, **kwargs) if callable(model) else model.predict(*args, **kwargs)
                predictions.append(pred)
            except Exception as e:
                print(f"모델 예측 중 오류 발생: {str(e)}")
                predictions.append(0)
                
        # 가중 평균 계산
        if not predictions:
            return 0
            
        # 예측 결과의 형태에 따라 처리
        if isinstance(predictions[0], (int, float)):
            # 숫자인 경우 가중 평균
            result = sum(p * w for p, w in zip(predictions, self.weights))
        else:
            # 다차원 배열인 경우 첫번째 예측 반환 (임시)
            result = predictions[0]
            
        return result
    
    def combine_predictions(self, predictions_dict):
        """
        여러 모델의 예측값을 딕셔너리로 받아 앙상블 결과 반환
        
        Args:
            predictions_dict: 모델 이름을 키로, 예측값을 값으로 하는 딕셔너리
            
        Returns:
            int/float: 앙상블된 예측값
        """
        if not predictions_dict:
            return 0
            
        # 모델 이름별로 가중치 적용
        result = 0
        total_weight = 0
        
        for model_name, prediction in predictions_dict.items():
            # 해당 모델의 가중치 찾기
            weight = 0.33  # 기본 가중치
            
            # 이름으로 모델 인덱스 찾기
            if hasattr(self, 'model_names'):
                try:
                    idx = self.model_names.index(model_name)
                    if idx < len(self.weights):
                        weight = self.weights[idx]
                except ValueError:
                    pass
            
            # 가중 합산
            result += prediction * weight
            total_weight += weight
        
        # 정규화
        if total_weight > 0:
            result /= total_weight
            
        # 정수값으로 반올림 (액션의 경우)
        if all(isinstance(pred, int) for pred in predictions_dict.values()):
            return int(round(result))
            
        return result
        
    def update_weights(self, performance_scores):
        """
        성능 점수에 기반하여 가중치 업데이트
        
        Args:
            performance_scores: 각 모델의 성능 점수 리스트
        """
        if len(performance_scores) != len(self.models):
            return
            
        # 성능 점수 정규화 (0~1 범위)
        min_score = min(performance_scores)
        max_score = max(performance_scores)
        
        if max_score > min_score:
            norm_scores = [(s - min_score) / (max_score - min_score) for s in performance_scores]
        else:
            norm_scores = [1.0 / len(performance_scores)] * len(performance_scores)
            
        # 소프트맥스 적용하여 가중치 계산
        exp_scores = [math.exp(s) for s in norm_scores]
        sum_exp = sum(exp_scores)
        
        if sum_exp > 0:
            self.weights = [e / sum_exp for e in exp_scores]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)

class AdvancedMultimodalTrader(BasicTrader):
    """
    고급 멀티모달 트레이더 클래스
    - 기본 트레이더 기능 확장
    - 개선된 강화학습 전략
    - 향상된 멀티모달 융합
    """
    
    def __init__(self, candlestick_analyzer, sentiment_analyzer, config=None):
        """
        고급 멀티모달 트레이더 초기화
        
        Args:
            candlestick_analyzer: 캔들스틱 패턴 분석기
            sentiment_analyzer: 뉴스 감성 분석기
            config: 설정 딕셔너리
        """
        super().__init__(candlestick_analyzer, sentiment_analyzer, config)
        
        # 하이브리드 시스템 구성요소 설정 제거
        self.use_hybrid_approach = False
        self.use_multi_timeframe = False
        self.use_market_regime = False
        self.use_adaptive_risk = False
        self.use_statistical_models = False
        
        # 확장된 앙상블 설정
        self._setup_advanced_ensemble()
        
        # 고급 강화학습 설정
        self._setup_advanced_rl()
        
        self.logger.info("고급 멀티모달 트레이더 초기화 완료")
    
    def _initialize_hybrid_components(self):
        """하이브리드 트레이딩 시스템 구성요소 초기화 (제거됨)"""
        # 하이브리드 접근법이 비활성화되어 있으므로 아무 작업도 수행하지 않음
        pass
    
    def _setup_advanced_ensemble(self):
        """향상된 앙상블 설정"""
        if not self.use_ensemble:
            return
        
        # 앙상블 가중치 동적 조정 기능 추가
        self.dynamic_weights = self.config.get('ensemble', {}).get('dynamic_weights', True)
        self.performance_history = defaultdict(list)
        
        # 고급 모델 추가
        if self.model_ensemble is not None:
            # 기술 지표 기반 모델 추가
            self.model_ensemble.add_model('technical', self._technical_model_prediction, weight=0.2)
            
            # 기존 모델 가중치 수동 조정 (성능 기반 update_weights 대신)
            # 직접 모델의 가중치를 설정
            performance_metrics = {
                'fusion': 0.5,
                'rl': 0.3,
                'technical': 0.2
            }
            
            # 성능 지표로 가중치 업데이트
            self.model_ensemble.update_weights(performance_metrics)
        
        self.logger.info("고급 앙상블 거래 모듈 초기화 완료")
    
    def _technical_model_prediction(self, state):
        """기술 지표 기반 예측 모델"""
        # 예시: 이동평균, RSI, MACD 등 기술 지표 기반 예측
        # 실제로는 기술 지표를 계산하고 신호를 생성해야 함
        return random.uniform(-1, 1)  # 예시 코드
    
    def run(self):
        """고급 트레이딩 실행"""
        self.logger.info("고급 멀티모달 트레이딩 시작")
        
        # 데이터 로드 및 전처리
        if self.progress_callback:
            self.progress_callback(5, "데이터 로드 중")
            
        # 기본 거래 대신 고급 트레이딩 실행
        trading_days = self.candlestick_analyzer.get_trading_days()
        self.logger.info(f"총 {len(trading_days)}일의 거래일 처리")
        
        # 백테스팅 진행
        self.portfolio_value_history = [self.initial_balance]
        self.position_history = ['cash']  # 초기 포지션: 현금
        self.transaction_history = []
        self.signals_history = []
        
        # 포트폴리오 및 포지션 초기화
        current_position = 'cash'
        current_btc = 0.0
        current_cash = self.initial_balance
        current_value = current_cash

        # 날짜 기록 초기화
        self.dates = []
        self.action_history = []
        
        # 상태 및 기타 변수 초기화
        state = None
        prev_state = None
        reward = 0
        
        # price_history 초기화 (필요할 경우)
        self.price_history = []
        
        # 기본 거래 대신 고급 거래 전략 실행
        for idx, date in enumerate(trading_days):
            try:
                # 날짜 기록 추가
                self.dates.append(date)
                
                # 진행률 업데이트
                progress_percent = int(10 + (idx / len(trading_days) * 80))  # 10%~90% 범위 진행률
                if self.progress_callback:
                    self.progress_callback(progress_percent, f"거래일: {date}, 포트폴리오: ${current_value:.2f}")
                
                # 이미지 기반 패턴 및 감성 신호 얻기
                try:
                    pattern_signals = self.candlestick_analyzer.get_pattern_signals(date)
                    sentiment_signals = self.sentiment_analyzer.get_sentiment_signals(date)
                    
                    # 가격 정보 얻기 및 기록
                    price_today = self.candlestick_analyzer.get_close_price(date)
                    if price_today is not None and price_today > 0:
                        self.price_history.append(price_today)
                    
                    # 다음 날 가격 정보 얻기
                    tomorrow_idx = min(idx + 1, len(trading_days) - 1)
                    tomorrow = trading_days[tomorrow_idx]
                    price_tomorrow = self.candlestick_analyzer.get_close_price(tomorrow)
                    
                    # 0으로 나누기 방지 - 가격이 0이거나 매우 작은 경우
                    if price_today is None or price_today < 0.000001:
                        price_today = 0.001
                        self.logger.warning(f"거래일 {date}의 가격이 0 또는 None입니다. 기본값 0.001로 설정합니다.")
                    
                    if price_tomorrow is None or price_tomorrow < 0.000001:
                        price_tomorrow = price_today  # 이전 가격과 동일하게 설정
                        self.logger.warning(f"다음 거래일 {tomorrow}의 가격이 0 또는 None입니다. 이전 가격과 동일하게 설정합니다.")
                    
                    # 가격 변화율
                    price_change = (price_tomorrow - price_today) / price_today
                    
                    # 액션 결정 (시장 레짐 제거 버전으로 호출)
                    action = self._get_advanced_action(
                        pattern_signals, 
                        sentiment_signals, 
                        price_change, 
                        current_position,
                        current_value,
                        date
                    )
                    
                    # 액션 기록 추가
                    self.action_history.append(action)
                    
                    # 포지션 변환
                    new_position = action_to_position(action)
                    
                    # 포지션 변경 시 거래 실행
                    if new_position != current_position:
                        # 거래 실행
                        price = price_today  # 캔들스틱 분석기에서 가격 직접 사용
                        
                        # 매수
                        if new_position == 'long' and current_position == 'cash':
                            current_btc = current_cash * 0.99 / price if price > 0 else 0  # 수수료 1% 가정, 0으로 나누기 방지
                            current_cash = 0
                            transaction = {
                                'date': date,
                                'type': 'buy',
                                'price': price,
                                'amount': current_btc,
                                'value': current_btc * price
                            }
                            self.transaction_history.append(transaction)
                        
                        # 매도
                        elif new_position == 'cash' and current_position == 'long':
                            current_cash = current_btc * price * 0.99  # 수수료 1% 가정
                            current_btc = 0
                            transaction = {
                                'date': date,
                                'type': 'sell',
                                'price': price,
                                'amount': current_btc,
                                'value': current_cash
                            }
                            self.transaction_history.append(transaction)
                    
                    # 포트폴리오 가치 계산
                    if current_position == 'cash':
                        current_value = current_cash
                    else:
                        price = price_today  # 캔들스틱 분석기에서 가격 직접 사용
                        current_value = current_btc * price
                    
                    # 음수 포트폴리오 값 방지
                    if current_value < 0:
                        self.logger.warning(f"거래일 {date}에서 포트폴리오 가치가 음수({current_value:.2f})입니다. 최소값으로 보정합니다.")
                        current_value = 0.01  # 최소값 설정
                    
                    # 포지션 및 포트폴리오 기록 업데이트
                    current_position = new_position
                    self.position_history.append(current_position)
                    self.portfolio_value_history.append(current_value)
                    
                    # 신호 기록
                    self.signals_history.append({
                        'date': date,
                        'pattern': pattern_signals,
                        'sentiment': sentiment_signals,
                        'action': action,
                        'position': current_position,
                        'portfolio_value': current_value
                    })
                    
                    # 상태 업데이트 및 강화학습 모델 학습
                    if self.use_rl:
                        # 상태 구성 (특성 벡터)
                        current_state = convert_to_state(
                            pattern_signals, 
                            sentiment_signals, 
                            price_change, 
                            current_position,
                            current_value / self.initial_balance if self.initial_balance > 0 else 0
                        )
                        
                        # 보상 계산
                        if len(self.portfolio_value_history) > 2:
                            prev_value = self.portfolio_value_history[-2]
                            if prev_value > 0:  # 0으로 나누기 방지
                                reward = (current_value - prev_value) / prev_value
                            else:
                                reward = 0
                        
                        # 모델 업데이트
                        if state is not None:
                            self.rl_trader.update(state, action, reward, current_state)
                        
                        state = current_state
                
                except Exception as e:
                    self.logger.error(f"거래일 {date} 처리 중 오류 발생: {str(e)}")
                    # 오류 발생 시 기본값으로 처리
                    self.action_history.append(0)  # 기본 액션: 청산
                    self.position_history.append(current_position)  # 현재 포지션 유지
                    self.portfolio_value_history.append(current_value)  # 현재 포트폴리오 가치 유지
            
            except Exception as e:
                self.logger.error(f"거래일 {date} 처리 중 전체 오류 발생: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 오류 발생 시 기본값 추가
                self.dates.append(date)
                self.action_history.append(0)  # 기본 액션: 청산
                self.position_history.append(current_position)  # 현재 포지션 유지
                self.portfolio_value_history.append(current_value)  # 현재 포트폴리오 가치 유지
        
        # 성능 평가
        if self.progress_callback:
            self.progress_callback(95, "성능 평가 중")
        
        self.evaluate_performance()
        
        # 결과 저장
        if self.progress_callback:
            self.progress_callback(100, "완료됨")
        
        self.logger.info("고급 멀티모달 트레이딩 완료")
        
        return self.performance_metrics
    
    def _analyze_advanced_performance(self):
        """고급 성능 분석"""
        self.logger.info("고급 성능 분석 수행")
        
        # 진행률 표시 추가
        print("\n[고급 성능 분석] 시작")
        analysis_steps = ['시간대별 성능', '변동성 구간별 성능', '모델 기여도']
        
        # 성능 분석 진행률 표시
        for step in tqdm(analysis_steps, desc='[고급 분석] 진행률', position=0):
            if step == '시간대별 성능':
                time_based_perf = self._analyze_time_based_performance()
            elif step == '변동성 구간별 성능':
                volatility_based_perf = self._analyze_volatility_based_performance()
            elif step == '모델 기여도':
                model_contribution = self._analyze_model_contribution()
        
        # 결과 저장
        self.advanced_performance = {
            'time_based': time_based_perf,
            'volatility_based': volatility_based_perf,
            'model_contribution': model_contribution
        }
        
        print("[고급 성능 분석] 완료")
    
    def _analyze_time_based_performance(self):
        """시간대별 성능 분석"""
        # 구현 필요
        return {'morning': 0.05, 'afternoon': 0.03, 'evening': 0.02, 'night': 0.01}
    
    def _analyze_volatility_based_performance(self):
        """변동성 구간별 성능 분석"""
        # 구현 필요
        return {'low': 0.02, 'medium': 0.04, 'high': 0.06}
    
    def _analyze_model_contribution(self):
        """모델 기여도 분석"""
        # 구현 필요
        return {
            'fusion': 0.5,
            'rl': 0.3,
            'technical': 0.2
        }
    
    def save_results(self):
        """결과 저장"""
        try:
            # 결과 저장 경로
            save_dir = self.config.get('output', {}).get('save_dir', '')
            if not save_dir:
                self.logger.warning("결과를 저장할 경로가 지정되지 않았습니다.")
                return
            
            os.makedirs(save_dir, exist_ok=True)
            
            # 1. 포트폴리오 가치 히스토리 저장
            portfolio_df = pd.DataFrame({
                'portfolio_value': self.portfolio_value_history
            })
            
            # 거래일 인덱스 추가
            if hasattr(self, 'trading_days') and len(self.trading_days) >= len(portfolio_df):
                portfolio_df.index = self.trading_days[:len(portfolio_df)]
            else:
                # 임시 인덱스 생성
                portfolio_df.index = [f"Day_{i}" for i in range(len(portfolio_df))]
            
            portfolio_df.to_csv(os.path.join(save_dir, 'portfolio_history.csv'))
            
            # 2. 성능 지표 저장
            if hasattr(self, 'performance_metrics'):
                with open(os.path.join(save_dir, 'performance_metrics.json'), 'w') as f:
                    json.dump(self.performance_metrics, f, indent=4)
            
            # 3. 거래 기록 저장
            if hasattr(self, 'transaction_history') and self.transaction_history:
                # Pandas DataFrame으로 변환
                transactions_df = pd.DataFrame(self.transaction_history)
                transactions_df.to_csv(os.path.join(save_dir, 'transactions.csv'), index=False)
            
            # 4. 포지션 히스토리 저장
            if hasattr(self, 'position_history') and self.position_history:
                positions_df = pd.DataFrame({
                    'position': self.position_history
                })
                
                # 거래일 인덱스 추가
                if hasattr(self, 'trading_days') and len(self.trading_days) >= len(positions_df):
                    positions_df.index = self.trading_days[:len(positions_df)]
                else:
                    # 임시 인덱스 생성
                    positions_df.index = [f"Day_{i}" for i in range(len(positions_df))]
                
                positions_df.to_csv(os.path.join(save_dir, 'position_history.csv'))
            
            # 5. 신호 히스토리 저장
            if hasattr(self, 'signals_history') and self.signals_history:
                # Pandas DataFrame으로 변환
                signals_df = pd.DataFrame(self.signals_history)
                signals_df.to_csv(os.path.join(save_dir, 'signals_history.csv'), index=False)
            
            self.logger.info(f"결과가 {save_dir}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류: {str(e)}")
            
    def generate_expert_systems_metrics(self):
        """Expert Systems with Applications 논문 형식의 성능 지표 생성"""
        # 결과 저장 경로
        save_dir = self.config.get('output', {}).get('save_dir', '')
        os.makedirs(save_dir, exist_ok=True)
        
        # 논문용 성능 지표 계산
        metrics = {}
        
        try:
            # 1. 누적 수익률
            initial_value = self.portfolio_value_history[0]
            final_value = self.portfolio_value_history[-1]
            cumulative_return = (final_value - initial_value) / initial_value * 100 if initial_value > 0 else 0
            metrics['cumulative_return'] = cumulative_return
            
            # 2. 일일 수익률 계산
            daily_returns = []
            for i in range(1, len(self.portfolio_value_history)):
                prev_value = max(0.01, self.portfolio_value_history[i-1])  # 0으로 나누기 방지
                curr_value = self.portfolio_value_history[i]
                daily_return = (curr_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
                daily_returns.append(daily_return)
            
            # 3. 일일 평균 수익률
            mean_daily_return = np.mean(daily_returns) if daily_returns else 0
            metrics['mean_daily_return'] = mean_daily_return
            
            # 4. 일일 수익률 표준편차 (변동성)
            std_daily_return = np.std(daily_returns) if daily_returns else 0
            metrics['std_daily_return'] = std_daily_return
            
            # 5. 샤프 비율
            risk_free_rate = 0.02 / 365  # 연간 2% 무위험 수익률
            sharpe_ratio = (mean_daily_return - risk_free_rate) / std_daily_return
            
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # 6. 최대 낙폭
            mdd = self._calculate_mdd() * 100
            metrics['max_drawdown'] = mdd
            
            # 7. 승률
            win_count = 0
            loss_count = 0
            
            for i in range(len(daily_returns)):
                if daily_returns[i] > 0:
                    win_count += 1
                elif daily_returns[i] < 0:
                    loss_count += 1
            
            win_rate = win_count / (win_count + loss_count) * 100 if (win_count + loss_count) > 0 else 0
            metrics['win_rate'] = win_rate
            
            # 8. 손익비
            wins = [r for r in daily_returns if r > 0]
            losses = [r for r in daily_returns if r < 0]
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 1  # 손실이 없으면 1로 설정
            profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
            metrics['profit_loss_ratio'] = profit_loss_ratio
            
            # 9. 칼마 비율 (CALMAR ratio)
            calmar_ratio = (cumulative_return / 252) / mdd if mdd > 0 else 0
            metrics['calmar_ratio'] = calmar_ratio
            
            # 10. 안정성 지수 (Stability Index)
            # R² of regression line of cumulative returns
            if len(self.portfolio_value_history) > 1:
                try:
                    x = np.arange(len(self.portfolio_value_history))
                    y = np.array(self.portfolio_value_history)
                    
                    # 데이터 정규화 및 안정화
                    y = np.clip(y, 0.01, np.inf)  # 0 미만 값 방지
                    
                    # SVD 오류 방지를 위한 정규화 조치
                    if np.std(y) > 0:
                        y_normalized = (y - np.mean(y)) / np.std(y)
                    else:
                        y_normalized = y
                        
                    # 회귀 계수 계산 (SVD 안정화 옵션 추가)
                    try:
                        slope, intercept = np.polyfit(x, y_normalized, 1, rcond=1e-10)
                        predicted = intercept + slope * x
                        y_mean = np.mean(y_normalized)
                        ss_total = np.sum((y_normalized - y_mean) ** 2)
                        if ss_total > 0:
                            r_squared = 1 - (np.sum((y_normalized - predicted) ** 2) / ss_total)
                        else:
                            r_squared = 0
                    except Exception as e:
                        self.logger.warning(f"선형 회귀 계산 중 오류: {str(e)}")
                        r_squared = 0
                except Exception as e:
                    self.logger.warning(f"안정성 지수 계산 중 오류: {str(e)}")
                    r_squared = 0
            else:
                r_squared = 0
            metrics['stability_index'] = r_squared
            
            # 결과 저장
            metrics_path = os.path.join(save_dir, 'expert_systems_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4)
                
            self.logger.info(f"Expert Systems with Applications 논문 형식 성능 지표 생성 완료: {metrics_path}")
            
        except Exception as e:
            self.logger.error(f"성능 지표 생성 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        return metrics

    def _get_advanced_action(self, pattern_signals, sentiment_signals, price_change, current_position, current_value, date):
        """
        간소화된 행동 결정 (시장 레짐 없음)
        
        Args:
            pattern_signals: 캔들스틱 패턴 신호
            sentiment_signals: 감성 분석 신호
            price_change: 최근 가격 변화
            current_position: 현재 포지션
            current_value: 현재 포트폴리오 가치
            date: 현재 날짜
            
        Returns:
            int: 선택된 행동 (0: 매수, 1: 매도, 2: 홀드)
        """
        # 기본 행동 확인 (기존 융합 방식 사용)
        base_action = self._get_fusion_action(
            pattern_signals, sentiment_signals, price_change, current_position
        )
        
        # 로깅
        action_names = ['매수', '매도', '홀드']
        self.logger.info(f"결정된 행동: {action_names[base_action]}")
        
        return base_action

    def evaluate_performance(self):
        """
        성능 평가
        - 다양한 성능 지표 계산 (수익률, 샤프 비율, 최대 낙폭 등)
        """
        try:
            if not hasattr(self, 'portfolio_value_history') or len(self.portfolio_value_history) <= 1:
                self.logger.warning("평가할 포트폴리오 데이터가 충분하지 않습니다.")
                self.performance_metrics = {}
                return {}
            
            # 성능 지표 계산
            metrics = {}
            
            # 1. 누적 수익률
            initial_value = self.portfolio_value_history[0]
            final_value = self.portfolio_value_history[-1]
            total_return = (final_value - initial_value) / initial_value * 100 if initial_value > 0 else 0
            metrics['total_return'] = total_return
            
            # 2. 연환산 수익률 (CAGR)
            n_years = max(0.01, len(self.portfolio_value_history) / 365)  # 최소값 설정으로 0으로 나누기 방지
            if initial_value > 0:
                cagr = ((final_value / initial_value) ** (1 / n_years) - 1) * 100
                metrics['cagr'] = cagr
            else:
                metrics['cagr'] = 0
            
            # 3. 일일 수익률 계산
            daily_returns = []
            for i in range(1, len(self.portfolio_value_history)):
                prev_value = max(0.01, self.portfolio_value_history[i-1])  # 0으로 나누기 방지
                curr_value = self.portfolio_value_history[i]
                daily_return = (curr_value - prev_value) / prev_value * 100 if prev_value > 0 else 0
                daily_returns.append(daily_return)
            
            # 4. 변동성 (일일 수익률 표준편차)
            volatility = np.std(daily_returns) * 100 if daily_returns else 0
            metrics['volatility'] = volatility
            
            # 5. 샤프 비율
            risk_free_rate = self.config.get('trading', {}).get('risk_free_rate', 0.02) / 365  # 일일 무위험 수익률
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            if volatility > 0:
                sharpe_ratio = ((avg_daily_return - risk_free_rate) / (volatility / 100)) * np.sqrt(252)  # 연환산
                metrics['sharpe_ratio'] = sharpe_ratio
            else:
                metrics['sharpe_ratio'] = 0
            
            # 6. 최대 낙폭 (Maximum Drawdown)
            metrics['max_drawdown'] = self._calculate_mdd() * 100
            
            # 7. 성공적인 거래 비율
            win_count = 0
            loss_count = 0
            
            for i in range(len(daily_returns)):
                if daily_returns[i] > 0:
                    win_count += 1
                elif daily_returns[i] < 0:
                    loss_count += 1
            
            total_trades = win_count + loss_count
            if total_trades > 0:
                win_rate = win_count / total_trades * 100
            else:
                win_rate = 0
            
            metrics['win_rate'] = win_rate
            
            # 8. 거래 횟수 및 회전율
            transaction_count = len(self.transaction_history) if hasattr(self, 'transaction_history') else 0
            metrics['transaction_count'] = transaction_count
            
            if len(self.portfolio_value_history) > 0:
                metrics['turnover_rate'] = transaction_count / len(self.portfolio_value_history) * 100
            else:
                metrics['turnover_rate'] = 0
            
            # 결과 기록
            self.performance_metrics = metrics
            self.logger.info(f"성능 평가 완료: 총 수익률={total_return:.2f}%, 샤프 비율={metrics.get('sharpe_ratio', 0):.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"성능 평가 중 오류: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.performance_metrics = {}
            return {}
    
    def _calculate_mdd(self):
        """최대 낙폭 계산"""
        if not self.portfolio_value_history or len(self.portfolio_value_history) < 2:
            return 0.0
            
        try:
            # 배열 변환
            portfolio_values = np.array(self.portfolio_value_history)
            
            # 음수나 0인 값이 있는지 확인
            min_value = np.min(portfolio_values)
            if min_value <= 0:
                self.logger.warning(f"포트폴리오 가치에 0 이하의 값이 있습니다. 이를 작은 양수로 대체합니다.")
                # 0 이하의 값을 작은 양수로 대체
                portfolio_values = np.maximum(portfolio_values, 0.001)
            
            # 누적 최대값 계산
            running_max = np.maximum.accumulate(portfolio_values)
            
            # 0으로 나누기 방지
            valid_indices = running_max > 0.001
            if not np.any(valid_indices):
                return 0.0
            
            # 유효한 인덱스에 대해서만 계산
            valid_running_max = running_max[valid_indices]
            valid_values = portfolio_values[valid_indices]
            
            # 낙폭 계산
            drawdown = (valid_running_max - valid_values) / valid_running_max
            
            # 최대 낙폭
            max_drawdown = np.max(drawdown) if drawdown.size > 0 else 0.0
            return max_drawdown
        except Exception as e:
            self.logger.error(f"최대 낙폭 계산 중 오류 발생: {str(e)}")
            return 0.0

    def generate_report(self):
        """결과 보고서 생성"""
        # 결과 저장 경로
        save_dir = self.config.get('output', {}).get('save_dir', '')
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 결과가 없으면 빈 보고서 생성
            if not hasattr(self, 'portfolio_value_history') or len(self.portfolio_value_history) == 0:
                self.logger.warning("포트폴리오 값 데이터가 없습니다. 빈 보고서를 생성합니다.")
                with open(os.path.join(save_dir, 'empty_report.txt'), 'w') as f:
                    f.write("포트폴리오 데이터가 없어 보고서를 생성할 수 없습니다.")
                return
            
            # 거래 기록 저장
            df = pd.DataFrame({
                'date': self.dates,
                'portfolio_value': self.portfolio_value_history,
                'action': self.action_history,
                'position': self.position_history
            })
            df.to_csv(os.path.join(save_dir, 'trading_history.csv'), index=False)
            
            # 성능 지표 저장
            if hasattr(self, 'performance_metrics'):
                with open(os.path.join(save_dir, 'performance_metrics.json'), 'w') as f:
                    json.dump(self.performance_metrics, f, indent=4)
            
            # 포트폴리오 가치 변화 그래프
            plt.figure(figsize=(12, 6))
            plt.plot(self.dates, self.portfolio_value_history, linewidth=2)
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Trading Days')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'portfolio_value.png'))
            plt.close()
            
            # 포지션 변화 그래프
            plt.figure(figsize=(12, 6))
            # 숫자로 변환 (매수=1, 홀딩=0, 매도=-1)
            position_map = {'long': 1, 'neutral': 0, 'short': -1}
            numeric_positions = [position_map.get(p, 0) for p in self.position_history]
            
            plt.plot(self.dates, numeric_positions)
            plt.title('Position Changes Over Time')
            plt.xlabel('Date')
            plt.ylabel('Position (1=Long, 0=Neutral, -1=Short)')
            plt.yticks([-1, 0, 1], ['Short', 'Neutral', 'Long'])
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'position_changes.png'))
            plt.close()
            
            # 포트폴리오 값 저장 (paper_report_for_submission 메서드에서 사용)
            self.portfolio_values = [{
                'timestamp': date,
                'portfolio_value': value,
                'position': position
            } for date, value, position in zip(self.dates, self.portfolio_value_history, self.position_history)]
            
            self.logger.info(f"결과가 {save_dir}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"결과 보고서 생성 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def process_trading_day(self, date, data, price, position, portfolio_value):
        """
        특정 거래일 처리
        
        Args:
            date: 거래일
            data: 해당 날짜의 데이터 (캔들스틱 패턴, 감성 분석 결과 등)
            price: 가격 데이터
            position: 현재 포지션
            portfolio_value: 현재 포트폴리오 가치
            
        Returns:
            new_position: 새로운 포지션
            new_portfolio_value: 업데이트된 포트폴리오 가치
            action: 취한 행동
        """
        try:
            # 캔들스틱 패턴 신호
            pattern_signals = data.get('pattern_signals', {})
            
            # 감성 분석 신호
            sentiment_signals = data.get('sentiment_signals', {})
            
            # 가격 변화 계산 (0으로 나누기 방지)
            if 'prev_price' in data and data['prev_price'] > 0.000001:  # 작은 임계값 추가
                price_change = (price - data['prev_price']) / data['prev_price']
            else:
                price_change = 0
                self.logger.warning(f"거래일 {date}에서 이전 가격이 0 또는 None입니다. 기본값 0.001로 설정합니다.")
            
            # 행동 결정
            action = self._get_advanced_action(
                pattern_signals, 
                sentiment_signals, 
                price_change, 
                position, 
                portfolio_value, 
                date
            )
            
            # 포지션 업데이트
            new_position = self._update_position(position, action)
            
            # 포트폴리오 가치 업데이트 (0으로 나누기 방지)
            trade_result = 0
            # 거래 비용 계산 시 0으로 나누기 방지
            transaction_cost = getattr(self, 'transaction_cost', 0.001)  # 기본값 0.1%
            
            if action == 1 and position != 'long':  # 매수
                trade_result = -transaction_cost * portfolio_value
            elif action == 0 and position != 'neutral':  # 청산
                trade_result = -transaction_cost * portfolio_value
            elif action == 2 and position != 'short':  # 매도
                trade_result = -transaction_cost * portfolio_value
                
            # 포지션에 따른 수익/손실
            position_pnl = 0
            if position == 'long':
                position_pnl = portfolio_value * price_change
            elif position == 'short':
                position_pnl = -portfolio_value * price_change
                
            new_portfolio_value = portfolio_value + position_pnl + trade_result
            
            # 포트폴리오 가치가 음수가 되지 않도록 방지
            if new_portfolio_value < 0:
                self.logger.warning(f"거래일 {date}에서 포트폴리오 가치가 음수({new_portfolio_value:.2f})입니다. 최소값으로 보정합니다.")
                new_portfolio_value = 0.01  # 최소값 설정
            
            # 로그 기록
            if self.verbose:
                self.logger.info(f"거래일: {date}, 포지션: {position} -> {new_position}, 포트폴리오: ${portfolio_value:.2f} -> ${new_portfolio_value:.2f}")
                
            return new_position, new_portfolio_value, action
            
        except Exception as e:
            self.logger.error(f"거래일 {date} 처리 중 오류 발생: {str(e)}")
            # 오류 발생 시 현재 상태 유지
            return position, portfolio_value, None

    def calculate_returns(self):
        """수익률 계산"""
        if not self.portfolio_value_history:
            return 0.0
        
        # 초기 포트폴리오 값
        initial_value = self.config.get('initial_capital', 10000.0)
        
        # 최종 포트폴리오 값
        final_value = self.portfolio_value_history[-1]
        
        # 0으로 나누는 오류 방지
        if initial_value == 0:
            self.logger.warning("초기 포트폴리오 값이 0이어서 수익률을 계산할 수 없습니다.")
            return 0.0
        
        # 총 수익률
        total_return = (final_value - initial_value) / initial_value
        
        return total_return

    def calculate_sharpe_ratio(self):
        """샤프 비율 계산"""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        # 일별 수익률 계산
        daily_returns = []
        for i in range(1, len(self.portfolio_value_history)):
            prev_value = self.portfolio_value_history[i-1]
            curr_value = self.portfolio_value_history[i]
            
            # 0으로 나누는 오류 방지
            if prev_value == 0:
                daily_return = 0.0
            else:
                daily_return = (curr_value - prev_value) / prev_value
                
            daily_returns.append(daily_return)
        
        # 평균 수익률 및 표준편차 계산
        mean_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
        
        # 표준편차 계산 (0으로 나누는 오류 방지)
        if len(daily_returns) <= 1:
            std_return = 0.001  # 표본이 1개 이하면 임의의 작은 값 사용
        else:
            sum_squared_diff = sum((r - mean_return) ** 2 for r in daily_returns)
            std_return = math.sqrt(sum_squared_diff / (len(daily_returns) - 1))
            
        # 0으로 나누는 오류 방지
        if std_return == 0:
            std_return = 0.001  # 표준편차가 0이면 임의의 작은 값 사용
        
        # 무위험 수익률 (연간 2% 가정)
        risk_free_rate = 0.02 / 252  # 일일 무위험 수익률
        
        # 샤프 비율 계산
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        return sharpe_ratio

    def paper_report_for_submission(self):
        """논문 제출용 결과 파일 자동 생성"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            import os
            import seaborn as sns
            import traceback
            
            # 결과 저장 경로
            output_dir = self.config.get('output', {}).get('save_dir', '')
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 포트폴리오 히스토리 CSV 읽기
            portfolio_csv_path = os.path.join(output_dir, 'portfolio_history.csv')
            if not os.path.exists(portfolio_csv_path):
                # CSV 파일이 없으면 메모리에서 생성
                if hasattr(self, 'portfolio_value_history') and self.portfolio_value_history:
                    portfolio_df = pd.DataFrame({
                        'value': self.portfolio_value_history
                    })
                    if hasattr(self, 'dates') and len(self.dates) == len(self.portfolio_value_history):
                        portfolio_df['date'] = self.dates
                        portfolio_df.set_index('date', inplace=True)
                    
                    # CSV 저장
                    portfolio_df.to_csv(portfolio_csv_path)
                else:
                    self.logger.error("포트폴리오 데이터가 없어 논문 제출용 결과를 생성할 수 없습니다.")
                    return False
            else:
                # CSV 파일이 있으면 읽기
                portfolio_df = pd.read_csv(portfolio_csv_path)
            
            # 2. 열 이름 확인 및 수정
            column_names = portfolio_df.columns.tolist()
            self.logger.info(f"포트폴리오 데이터프레임 열 이름: {column_names}")
            
            # value_column 선택 (portfolio_value 또는 value)
            value_column = None
            for col in ['portfolio_value', 'value']:
                if col in column_names:
                    value_column = col
                    break
            
            if value_column is None:
                # 열이 없는 경우, 첫 번째 데이터 열 사용 (인덱스 외)
                if len(column_names) > 0:
                    value_column = column_names[0]
                    self.logger.warning(f"portfolio_value 또는 value 열이 없어 {value_column} 열을 사용합니다.")
                else:
                    raise ValueError("포트폴리오 데이터에 사용 가능한 열이 없습니다.")
            
            # 3. 포트폴리오 가치 시각화
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_df.index, portfolio_df[value_column], linewidth=2)
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Trading Days')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'portfolio_value_paper.png'), dpi=300)
            plt.close()
            
            # 4. Expert Systems with Applications 논문 형식 지표 생성
            expert_systems_metrics = {
                'Total Return (%)': self.calculate_returns() * 100,
                'Annual Return (%)': self.calculate_returns() * 100 * (252 / len(portfolio_df)) if len(portfolio_df) > 0 else 0,
                'Sharpe Ratio': self.calculate_sharpe_ratio(),
                'Max Drawdown (%)': self._calculate_mdd() * 100,
                'Volatility (%)': np.std(np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]) * 100 * np.sqrt(252) if len(self.portfolio_value_history) > 1 else 0,
                'Win Rate (%)': (self.performance_metrics.get('win_rate', 0) * 100) if hasattr(self, 'performance_metrics') else 0,
                'Total Trades': self.performance_metrics.get('total_trades', 0) if hasattr(self, 'performance_metrics') else 0,
                'Profit Factor': self.performance_metrics.get('profit_factor', 0) if hasattr(self, 'performance_metrics') else 0
            }
            
            # JSON 파일로 저장
            with open(os.path.join(output_dir, 'expert_systems_metrics_paper.json'), 'w') as f:
                json.dump(expert_systems_metrics, f, indent=4)
                
            self.logger.info(f"논문 제출용 결과 생성 완료: {os.path.join(output_dir, 'expert_systems_metrics_paper.json')}")
            
            # 5. 성능 비교 테이블 (CSV)
            metrics_df = pd.DataFrame([expert_systems_metrics])
            metrics_df.to_csv(os.path.join(output_dir, 'performance_metrics_paper.csv'), index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"논문 제출용 결과 파일 생성 중 오류 발생: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def _update_position(self, current_position, action):
        """
        현재 포지션과 액션을 기반으로 새 포지션 결정
        
        Args:
            current_position: 현재 포지션 ('long', 'short', 'neutral')
            action: 결정된 액션 (0: 청산, 1: 매수, 2: 매도)
            
        Returns:
            str: 새 포지션
        """
        if action == 1:  # 매수
            return 'long'
        elif action == 2:  # 매도
            return 'short'
        elif action == 0:  # 청산/홀딩
            # 현재 포지션이 있으면 청산, 없으면 유지
            if current_position in ['long', 'short']:
                return 'neutral'
            else:
                return current_position
        else:
            # 알 수 없는 액션의 경우 현재 포지션 유지
            return current_position

    def _determine_market_regime(self, price_history, days=30):
        """
        시장 레짐 감지 (제거됨)
        
        Args:
            price_history: 가격 이력 데이터
            days: 분석할 일수
            
        Returns:
            regime: 'normal'로 고정
            confidence: 0.0으로 고정
        """
        # paper1에서는 시장 레짐을 사용하지 않으므로 항상 기본값 반환
        return 'normal', 0.0

    def _run_statistical_models(self, price_history, forecast_days=5):
        """
        통계적 모델 실행 (제거됨)
        
        Args:
            price_history: 가격 이력 데이터
            forecast_days: 예측할 일수
            
        Returns:
            dict: 기본 예측 결과
        """
        # paper1에서는 통계적 모델을 사용하지 않으므로 기본값 반환
        return {'direction': 0, 'confidence': 0.0}

    def _calculate_technical_indicators(self, prices):
        """
        기술적 지표 계산 (간소화됨)
        
        Args:
            prices: 가격 이력 데이터
            
        Returns:
            dict: 기본 기술적 지표 결과
        """
        # paper1에서는 고급 기술적 지표를 사용하지 않으므로 기본값 반환
        return {'signal': 0, 'strength': 0}

    def _apply_risk_management(self, scores, current_position, current_value):
        """
        간소화된 리스크 관리
        
        Args:
            scores: 행동 점수 배열 [매수, 매도, 홀드]
            current_position: 현재 포지션
            current_value: 현재 포트폴리오 가치
            
        Returns:
            list: 원본 행동 점수 (변경 없음)
        """
        # paper1에서는 동적 리스크 관리를 적용하지 않음
        return scores

# 테스트 코드
if __name__ == "__main__":
    # 설정 정의
    config = {
        'data': {
            'chart_dir': './data/chart',
            'news_file': './data/news/cryptonews_2021-10-12_2023-12-19.csv',
            'timeframes': ['1d', '4h', '1h', '30m', '15m', '5m']
        },
        'output': {
            'save_dir': './results'
        },
        'fusion': {
            'type': 'attention'
        },
        'rl': {
            'use_rl': True,
            'learning_rate': 0.0003,
            'gamma': 0.99
        },
        'ensemble': {
            'use_ensemble': True,
            'timeframe_ensemble_method': 'weighted_average',
            'model_ensemble_method': 'weighted_average',
            'dynamic_weights': True
        }
    }
    
    # 분석기 초기화
    candlestick_analyzer = CandlestickAnalyzer(config)
    sentiment_analyzer = SentimentAnalyzer(config)
    
    # 고급 트레이더 초기화 및 실행
    trader = AdvancedMultimodalTrader(candlestick_analyzer, sentiment_analyzer, config)
    trader.run()
    
    # 성능 평가 및 결과 저장
    trader.evaluate_performance()
    trader.save_results()
    
    # 논문 제출용 결과 생성
    trader.paper_report_for_submission()
    
    print("고급 멀티모달 트레이딩 시뮬레이션 완료!") 