#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
트레이딩 전략 모듈
- 멀티모달 융합 결과를 기반으로 트레이딩 신호 생성
- 상승/보합/하락 예측에 따른 매수/매도/홀드 결정
- 멀티 타임프레임 분석 및 시장 레짐 인식 기능 추가
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TradingStrategy')

class FCNPolicy(nn.Module):
    """완전 연결 네트워크 기반 정책 네트워크"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super(FCNPolicy, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # 소프트맥스를 통해 각 행동(매수, 홀드, 매도)의 확률 계산
        action_probs = F.softmax(self.fc_out(x), dim=1)
        
        return action_probs

class MarketRegimeClassifier:
    """시장 레짐 분류기 - paper1에서는 사용하지 않음"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.regimes = ['normal']  # 단일 레짐만 사용
    
    def classify(self, prices):
        """
        가격 데이터 기반 시장 레짐 분류 (paper1에서는 항상 'normal' 반환)
        
        Args:
            prices: 가격 시계열 데이터
            
        Returns:
            regime: 기본 레짐 'normal'
            confidence: 기본 신뢰도 0.0
        """
        return 'normal', 0.0

class MultiTimeframeAnalyzer:
    """다중 시간프레임 분석기 - paper1에서는 단순 기능만 제공"""
    
    def __init__(self, timeframes=None, weights=None):
        # 기본 시간프레임 및 가중치 설정
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # 가중치 설정 (동일 가중치)
        self.weights = {}
        for tf in self.timeframes:
            self.weights[tf] = 1.0 / len(self.timeframes)
            
        # 시간프레임별 데이터 저장
        self.data = {tf: [] for tf in self.timeframes}
        
        # 시간프레임별 신호 저장
        self.signals = {tf: 0 for tf in self.timeframes}
    
    def update_data(self, timeframe, prices):
        """특정 시간프레임 데이터 업데이트"""
        if timeframe in self.timeframes:
            self.data[timeframe] = prices
    
    def update_signal(self, timeframe, signal):
        """특정 시간프레임 신호 업데이트"""
        if timeframe in self.timeframes:
            self.signals[timeframe] = signal
    
    def get_integrated_signal(self):
        """
        여러 시간프레임 신호 통합 (단순 평균)
        
        Returns:
            float: 통합 신호 (-1 ~ +1 범위, -1: 강한 매도, +1: 강한 매수)
        """
        if not self.signals:
            return 0.0
        
        # 단순 평균 계산
        signals_sum = sum(self.signals.values())
        signals_count = len(self.signals)
        
        if signals_count == 0:
            return 0.0
        
        # 통합 신호 (-1 ~ +1 범위)
        return signals_sum / signals_count
    
    def adjust_weights_by_regime(self, market_regime):
        """
        시장 레짐에 따라 시간프레임 가중치 조정 (paper1에서는 작동하지 않음)
        
        Args:
            market_regime: 현재 시장 레짐
        """
        # paper1에서는 시장 레짐에 따른 조정을 하지 않음
        pass

class TradingStrategy:
    """트레이딩 전략 클래스"""
    
    def __init__(self, config=None):
        """초기화"""
        self.config = config or {}
        
        # 로거 설정
        self.logger = logging.getLogger('TradingStrategy')
        
        # 결과 저장 경로
        output_config = self.config.get('output', {})
        self.results_dir = output_config.get('save_dir', 'results/paper1')
        
        # 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 정책 네트워크 구성
        self.policy_model = None
        
        # 차원 감소를 위한 PCA
        self.pca = None
        
        # 임계값 설정
        self.buy_threshold = self.config.get('buy_threshold', 0.6)
        self.sell_threshold = self.config.get('sell_threshold', 0.6)
        
        # paper1에서는 하이브리드 시스템 비활성화
        self.use_market_regime = False
        self.use_multi_timeframe = False
        
        self.logger.info("트레이딩 전략 초기화 완료")
    
    def _initialize_policy_network(self, input_dim):
        """정책 네트워크 초기화"""
        # GPU 사용 여부 확인
        use_gpu = self.config.get('use_gpu', True) and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_gpu else 'cpu')
        
        # 정책 네트워크 초기화
        self.policy_model = FCNPolicy(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=3  # 매수(1), 홀드(0), 매도(-1)
        ).to(self.device)
    
    def _apply_pca_reduction(self, features):
        """PCA 차원 축소 적용"""
        if self.pca is None:
            # PCA 초기화
            n_components = min(64, features.shape[1])
            self.pca = PCA(n_components=n_components)
            self.pca.fit(features)
        
        # PCA 적용
        return self.pca.transform(features)
    
    def update_market_regime(self, prices):
        """
        현재 시장 레짐 업데이트 (paper1에서는 사용하지 않음)
        
        Args:
            prices: 가격 시계열 데이터
        """
        # paper1에서는 시장 레짐을 사용하지 않음
        pass
    
    def update_timeframe_data(self, timeframe, prices):
        """
        특정 시간프레임의 가격 데이터 업데이트
        
        Args:
            timeframe: 시간프레임 ('1m', '5m', '15m', '1h', '4h', '1d')
            prices: 가격 시계열 데이터
        """
        if not self.use_multi_timeframe:
            return
        
        self.timeframe_analyzer.update_data(timeframe, prices)
    
    def generate_signals(self, features, prices=None):
        """융합된 특징 벡터로부터 트레이딩 신호 생성"""
        self.logger.info(f"트레이딩 신호 생성 중... ({len(features)} 예측)")
        
        try:
            # 특징 벡터가 비어있는지 확인
            if len(features) == 0:
                return []
            
            # 시장 레짐 업데이트 (가격 데이터가 있는 경우)
            if prices is not None and self.use_market_regime:
                self.update_market_regime(prices)
            
            # 입력 차원 확인
            input_dim = features.shape[1]
            
            # 정책 네트워크 초기화
            if self.policy_model is None:
                self._initialize_policy_network(input_dim)
            
            # 차원 축소 적용 (복잡도 감소)
            # features = self._apply_pca_reduction(features)
            
            # 특징 벡터를 텐서로 변환
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # 각 특징 벡터에 대한 신호 생성
            signals = []
            probabilities = []
            
            # 미니 배치로 처리
            batch_size = 32
            
            self.policy_model.eval()  # 평가 모드로 설정
            
            with torch.no_grad():
                for i in range(0, len(features_tensor), batch_size):
                    batch = features_tensor[i:i+batch_size]
                    
                    # 모의 정책 네트워크 출력 (실제로는 학습된 모델 사용)
                    # action_probs = self.policy_model(batch)
                    
                    # 더미 정책 네트워크 출력 (랜덤 생성)
                    action_probs = self._generate_simulated_probabilities(batch)
                    
                    # 확률에 기반하여 행동 결정
                    actions, probs = self._decide_actions(action_probs)
                    
                    signals.extend(actions)
                    probabilities.extend(probs)
            
            # 시장 레짐 기반 신호 조정
            if self.use_market_regime:
                signals = self._adjust_signals_by_regime(signals, probabilities)
            
            # 멀티 타임프레임 신호 통합
            if self.use_multi_timeframe and len(signals) > 0:
                # 대표 신호를 멀티 타임프레임 분석기에 추가
                self.timeframe_analyzer.update_signal('1m', signals[0])  # 예시로 1m 시간프레임 사용
                
                # 통합 신호 가져오기
                integrated_signal = self.timeframe_analyzer.get_integrated_signal()
                
                # 필요 시 통합 신호를 반영하여 최종 신호 조정
                if abs(integrated_signal) > 0.6:  # 강한 신호인 경우
                    modified_signal = 1 if integrated_signal > 0 else -1
                    signals[0] = modified_signal
            
            # 신호 통계
            signal_counts = {
                1: sum(1 for s in signals if s == 1),  # 매수
                0: sum(1 for s in signals if s == 0),  # 홀드
                -1: sum(1 for s in signals if s == -1)  # 매도
            }
            
            self.logger.info(f"트레이딩 신호 생성 완료: 매수 {signal_counts[1]}건, 홀드 {signal_counts[0]}건, 매도 {signal_counts[-1]}건")
            
            return signals
        
        except Exception as e:
            self.logger.exception(f"신호 생성 중 오류 발생: {str(e)}")
            return [0] * len(features)  # 오류 발생 시 모두 홀드로 설정
    
    def _adjust_signals_by_regime(self, signals, probabilities):
        """
        시장 레짐에 기반한 신호 조정 (paper1에서는 원본 신호 반환)
        
        Args:
            signals: 원본 신호 리스트
            probabilities: 신호별 확률값
            
        Returns:
            adjusted_signals: 원본 신호 리스트 (변경 없음)
        """
        # paper1에서는 시장 레짐에 따른 조정을 하지 않음
        return signals
    
    def _generate_simulated_probabilities(self, features_batch):
        """
        시뮬레이션된 확률 생성
        
        Args:
            features_batch (torch.Tensor): 특징 배치
            
        Returns:
            torch.Tensor: 각 행동에 대한 확률 (매수, 홀드, 매도)
        """
        batch_size = features_batch.size(0)
        
        # 특징 벡터의 첫 번째 차원을 기준으로 확률 분포 결정
        # 예: 첫 번째 특징이 0.7보다 크면 매수 확률이 높아짐
        buy_prob = torch.zeros(batch_size, 1)
        hold_prob = torch.zeros(batch_size, 1)
        sell_prob = torch.zeros(batch_size, 1)
        
        # 시드 고정
        torch.manual_seed(42)
        
        # 특징 합계를 계산하여 신호 결정
        feature_sums = torch.sum(features_batch, dim=1)
        feature_means = torch.mean(features_batch, dim=1)
        feature_stds = torch.std(features_batch, dim=1)
        
        # 정규화
        norm_sums = (feature_sums - torch.mean(feature_sums)) / (torch.std(feature_sums) + 1e-8)
        
        for i in range(batch_size):
            # 트렌드 특성 추출 (예시)
            trend_indicator = norm_sums[i]
            
            # 변동성 지표
            volatility = feature_stds[i]
            
            # 매수 신호: 트렌드가 상승 중이고 변동성이 낮은 경우
            if trend_indicator > 0.5 and volatility < torch.median(feature_stds):
                buy_prob[i] = 0.7 + torch.rand(1) * 0.2
                hold_prob[i] = 0.8 - buy_prob[i]
                sell_prob[i] = 0.2 - hold_prob[i]
            
            # 매도 신호: 트렌드가 하락 중이고 변동성이 낮은 경우
            elif trend_indicator < -0.5 and volatility < torch.median(feature_stds):
                sell_prob[i] = 0.7 + torch.rand(1) * 0.2
                hold_prob[i] = 0.8 - sell_prob[i]
                buy_prob[i] = 0.2 - hold_prob[i]
            
            # 변동성이 높은 경우 홀드
            elif volatility > torch.median(feature_stds) * 1.5:
                hold_prob[i] = 0.8 + torch.rand(1) * 0.2
                buy_prob[i] = (1 - hold_prob[i]) / 2
                sell_prob[i] = (1 - hold_prob[i]) / 2
            
            # 기타 경우 (확률 무작위 생성)
            else:
                # 트렌드 방향에 따라 조금 더 높은 확률 부여
                if trend_indicator > 0:
                    buy_weight = 0.4
                    sell_weight = 0.2
                else:
                    buy_weight = 0.2
                    sell_weight = 0.4
                
                hold_weight = 1.0 - buy_weight - sell_weight
                
                # 조금 더 랜덤화
                buy_weight += (torch.rand(1) * 0.2 - 0.1).item()
                sell_weight += (torch.rand(1) * 0.2 - 0.1).item()
                hold_weight += (torch.rand(1) * 0.2 - 0.1).item()
                
                # 음수 방지 및 정규화
                buy_weight = max(0.1, buy_weight)
                sell_weight = max(0.1, sell_weight)
                hold_weight = max(0.1, hold_weight)
                
                total = buy_weight + sell_weight + hold_weight
                buy_prob[i] = buy_weight / total
                sell_prob[i] = sell_weight / total
                hold_prob[i] = hold_weight / total
        
        # 모든 행동 확률을 하나의 텐서로 결합
        action_probs = torch.cat([buy_prob, hold_prob, sell_prob], dim=1)
        
        return action_probs
    
    def _decide_actions(self, action_probs):
        """
        행동 확률을 바탕으로 실제 행동 결정
        
        Args:
            action_probs (torch.Tensor): 각 행동에 대한 확률
            
        Returns:
            list: 결정된 행동 (-1, 0, 1)
            list: 행동별 확률
        """
        actions = []
        probs = []
        
        for i in range(action_probs.size(0)):
            probs_arr = action_probs[i].cpu().numpy()
            
            # 가장 높은 확률을 가진 행동 선택
            action_idx = np.argmax(probs_arr)
            prob = probs_arr[action_idx]
            
            # 인덱스를 실제 행동으로 변환 (0: 매수, 1: 홀드, 2: 매도)
            if action_idx == 0:
                action = 1  # 매수
            elif action_idx == 1:
                action = 0  # 홀드
            else:
                action = -1  # 매도
            
            actions.append(action)
            probs.append(prob)
        
        return actions, probs
    
    def apply_risk_management(self, signals, prices):
        """
        위험 관리 로직 적용
        
        Args:
            signals: 원본 신호 리스트
            prices: 가격 데이터
            
        Returns:
            signals: 위험 관리가 적용된 신호 리스트
        """
        if len(signals) == 0 or prices is None:
            return signals
        
        # 시장 레짐 확인
        if self.use_market_regime:
            # 변동성 높은 시장에서 위험 관리 강화
            if self.current_regime == 'volatile' and self.regime_confidence > 0.7:
                # 포지션 크기 축소 또는 일부 신호 필터링 등의 로직 적용
                for i in range(len(signals)):
                    # 높은 변동성 시장에서는 무작위 신호 중 일부 필터링
                    if random.random() < 0.3:
                        signals[i] = 0  # 홀드로 변경
        
        # 연속 손실 방지 로직 등 추가 가능...
        
        return signals 