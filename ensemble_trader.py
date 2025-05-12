import torch
import numpy as np
import pandas as pd
import os
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class MultiTimeframeEnsemble:
    """
    다중 시간프레임 앙상블 모듈
    - 여러 시간프레임(5m, 15m, 30m, 1h, 4h, 1d)의 신호를 통합
    - 투표, 가중평균, 또는 다양한 앙상블 기법 적용
    """
    def __init__(self, timeframes=['5m', '15m', '30m', '1h', '4h', '1d'], 
                 weights=None, method='weighted_average'):
        self.timeframes = timeframes
        self.method = method  # 'voting', 'weighted_average', 'boosting'
        
        # 시간프레임별 가중치 설정 (기본: 장기간 시간프레임에 더 높은 가중치)
        if weights is None:
            # 시간프레임 길이에 비례하는 가중치 설정
            weights = {}
            base_weights = {
                '5m': 1.0,
                '15m': 1.5,
                '30m': 2.0,
                '1h': 2.5,
                '4h': 3.0,
                '1d': 4.0
            }
            total_weight = sum(base_weights[tf] for tf in timeframes if tf in base_weights)
            for tf in timeframes:
                if tf in base_weights:
                    weights[tf] = base_weights[tf] / total_weight
        
        self.weights = weights
        
        # 로깅 설정
        self.logger = logging.getLogger('MultiTimeframeEnsemble')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.logger.info(f"다중 시간프레임 앙상블 초기화: 방법={method}, 시간프레임={timeframes}")
        self.logger.info(f"시간프레임 가중치: {weights}")
    
    def combine_signals(self, signals):
        """
        여러 시간프레임의 신호를 통합
        
        Args:
            signals: dict - {timeframe: signal_value} 형태의 딕셔너리
                     signal_value는 -1.0 ~ 1.0 사이의 값 (-1: 강한 매도, 1: 강한 매수)
        
        Returns:
            float: 통합된 신호 (-1.0 ~ 1.0)
        """
        if not signals:
            return 0.0
        
        available_timeframes = [tf for tf in signals.keys() if tf in self.timeframes]
        if not available_timeframes:
            return 0.0
        
        if self.method == 'voting':
            # 투표 방식: 매수/매도/중립 카운트
            buy_votes = sum(1 for tf in available_timeframes if signals[tf] > 0.2)
            sell_votes = sum(1 for tf in available_timeframes if signals[tf] < -0.2)
            
            if buy_votes > sell_votes:
                # 매수 투표가 많으면 양수 신호
                return 0.3 + 0.7 * (buy_votes / len(available_timeframes))
            elif sell_votes > buy_votes:
                # 매도 투표가 많으면 음수 신호
                return -0.3 - 0.7 * (sell_votes / len(available_timeframes))
            else:
                # 동점이면 중립
                return 0.0
        
        elif self.method == 'weighted_average':
            # 가중 평균: 각 시간프레임에 가중치 적용
            total_weight = 0
            weighted_sum = 0
            
            for tf in available_timeframes:
                if tf in self.weights:
                    weight = self.weights[tf]
                    weighted_sum += signals[tf] * weight
                    total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.method == 'boosting':
            # 부스팅: 최근 성과가 좋은 시간프레임에 더 높은 가중치 부여
            # (이 예시에서는 단순화를 위해 가중 평균과 동일하게 구현)
            return self.combine_signals(signals, method='weighted_average')
        
        else:
            # 기본 방식: 단순 평균
            return sum(signals[tf] for tf in available_timeframes) / len(available_timeframes)
    
    def update_weights(self, performance_metrics):
        """
        성능 지표에 따라 시간프레임 가중치 동적 업데이트 (부스팅)
        
        Args:
            performance_metrics: dict - {timeframe: performance_score} 형태의 딕셔너리
        """
        if self.method != 'boosting':
            return
        
        # 성능 지표 정규화
        total_perf = sum(max(0.1, score) for score in performance_metrics.values())
        
        # 가중치 업데이트
        for tf in self.timeframes:
            if tf in performance_metrics:
                # 성능이 좋을수록 가중치 증가
                self.weights[tf] = max(0.1, performance_metrics[tf]) / total_perf
        
        self.logger.info(f"시간프레임 가중치 업데이트: {self.weights}")

class ModelEnsemble:
    """
    다양한 모델 앙상블 모듈
    - CNN, LSTM, Transformer 등 다양한 모델의 예측을 통합
    - 투표, 가중평균, 스태킹 등 다양한 앙상블 기법 적용
    - 시장 레짐에 따른 동적 가중치 조정 추가
    - 통계적 모델 통합 기능 추가
    """
    def __init__(self, models=None, model_weights=None, method='weighted_average'):
        self.models = models if models else {}  # 모델 딕셔너리: {model_name: model_obj}
        self.method = method  # 'voting', 'weighted_average', 'stacking'
        
        # 모델별 가중치 (기본: 동일 가중치)
        if model_weights is None and models:
            model_weights = {name: 1.0 / len(models) for name in models.keys()}
        
        self.weights = model_weights if model_weights else {}
        
        # 스태킹용 메타 모델 (단순 선형 모델)
        self.meta_model = torch.nn.Linear(len(self.models) if self.models else 1, 1)
        self.meta_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)
        
        # 시장 레짐별 모델 성과 기록
        self.regime_performance = {
            'trending': {},
            'volatile': {},
            'ranging': {},
            'normal': {}
        }
        
        # 현재 시장 레짐
        self.current_regime = 'normal'
        
        # 통계적 모델 통합
        self.use_statistical_models = False
        self.statistical_models = {}
        
        # 로깅 설정
        self.logger = logging.getLogger('ModelEnsemble')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.logger.info(f"모델 앙상블 초기화: 방법={method}, 모델 수={len(self.models) if self.models else 0}")
        if models:
            self.logger.info(f"모델 가중치: {self.weights}")
    
    def enable_statistical_models(self):
        """통계적 모델 활성화"""
        try:
            import statsmodels
            import arch
            
            self.use_statistical_models = True
            
            # 통계적 모델 초기화
            self.statistical_models = {
                'arima': {
                    'order': (2, 1, 0),
                    'weight': 0.3,
                    'model': None
                },
                'garch': {
                    'order': (1, 1),
                    'weight': 0.3,
                    'model': None
                }
            }
            
            self.logger.info("통계적 모델 활성화 성공")
            return True
        except ImportError:
            self.logger.warning("통계적 모델 활성화 실패: statsmodels 또는 arch 패키지가 필요합니다")
            return False
    
    def update_regime(self, price_history, regime='normal'):
        """
        시장 레짐 업데이트 및 가중치 조정
        
        Args:
            price_history: 가격 이력 데이터
            regime: 현재 시장 레짐 (직접 전달하거나 판단)
        """
        # 필요 시 자체적으로 레짐 판단
        if regime == 'auto' and len(price_history) >= 30:
            regime = self._detect_market_regime(price_history)
        
        # 레짐 업데이트
        prev_regime = self.current_regime
        self.current_regime = regime
        
        # 레짐이 변경된 경우 가중치 조정
        if regime != prev_regime:
            self._adjust_weights_for_regime(regime)
            self.logger.info(f"시장 레짐 변경: {prev_regime} -> {regime}, 가중치 조정됨")
    
    def _detect_market_regime(self, price_history):
        """
        가격 데이터로 시장 레짐 감지
        
        Args:
            price_history: 가격 이력 데이터
        
        Returns:
            str: 감지된 시장 레짐
        """
        if len(price_history) < 30:
            return 'normal'
        
        # 최근 가격 추출
        recent_prices = price_history[-30:]
        
        # 수익률 계산
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # 변동성 계산
        volatility = np.std(returns) * np.sqrt(252)  # 연간화
        
        # 방향성 계산 (양수 수익률 비율)
        direction = np.sum(returns > 0) / len(returns)
        
        # 추세 강도
        trend_strength = abs(direction - 0.5) * 2  # 0~1 범위로 정규화
        
        # 레짐 분류
        if volatility > 0.03:  # 높은 변동성
            return 'volatile'
        elif trend_strength > 0.6:  # 강한 추세
            return 'trending'
        elif volatility < 0.01 and trend_strength < 0.3:  # 낮은 변동성, 약한 추세
            return 'ranging'
        else:
            return 'normal'
    
    def _adjust_weights_for_regime(self, regime):
        """
        시장 레짐에 따른 모델 가중치 조정
        
        Args:
            regime: 현재 시장 레짐
        """
        # 해당 레짐에서의 성과 기록이 있는 경우
        if regime in self.regime_performance and self.regime_performance[regime]:
            perf = self.regime_performance[regime]
            
            # 성과에 기반한 가중치 계산
            total_perf = sum(max(0.1, p) for p in perf.values())
            
            if total_perf > 0:
                for model_name in self.weights:
                    if model_name in perf:
                        # 해당 레짐에서의 성과 기반 가중치
                        self.weights[model_name] = max(0.1, perf[model_name]) / total_perf
            
        else:
            # 레짐별 기본 가중치 설정
            if regime == 'volatile':
                # 변동성 높은 시장에서는 앙상블과 통계적 모델 선호
                for model_name in self.weights:
                    if 'ensemble' in model_name.lower():
                        self.weights[model_name] = 0.3
                    elif 'stat' in model_name.lower():
                        self.weights[model_name] = 0.25
                    elif 'lstm' in model_name.lower():
                        self.weights[model_name] = 0.2
                    else:
                        self.weights[model_name] = 0.1
            
            elif regime == 'trending':
                # 추세 시장에서는 LSTM과 CNN 모델 선호
                for model_name in self.weights:
                    if 'lstm' in model_name.lower():
                        self.weights[model_name] = 0.3
                    elif 'cnn' in model_name.lower():
                        self.weights[model_name] = 0.25
                    else:
                        self.weights[model_name] = 0.15
            
            elif regime == 'ranging':
                # 횡보장에서는 패턴 인식 모델 선호
                for model_name in self.weights:
                    if 'pattern' in model_name.lower():
                        self.weights[model_name] = 0.3
                    elif 'stat' in model_name.lower():
                        self.weights[model_name] = 0.25
                    else:
                        self.weights[model_name] = 0.15
        
        # 가중치 정규화
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for model_name in self.weights:
                self.weights[model_name] /= total_weight
    
    def run_statistical_models(self, price_history, forecast_days=5):
        """
        통계적 모델 실행
        
        Args:
            price_history: 가격 이력 데이터
            forecast_days: 예측 일수
            
        Returns:
            dict: 모델별 예측 결과
        """
        if not self.use_statistical_models or len(price_history) < 60:
            return {}
        
        predictions = {}
        
        try:
            # ARIMA 모델
            if 'arima' in self.statistical_models:
                from statsmodels.tsa.arima.model import ARIMA
                
                prices = np.array(price_history[-60:])
                model = ARIMA(prices, order=self.statistical_models['arima']['order'])
                result = model.fit()
                
                forecast = result.forecast(steps=forecast_days)
                direction = 1 if forecast[-1] > prices[-1] else (-1 if forecast[-1] < prices[-1] else 0)
                
                predictions['arima'] = {
                    'forecast': forecast.tolist(),
                    'direction': direction,
                    'confidence': 0.7  # 간단한 고정 신뢰도
                }
                
                # 모델 저장
                self.statistical_models['arima']['model'] = result
            
            # GARCH 모델
            if 'garch' in self.statistical_models:
                import arch
                
                returns = np.diff(price_history[-60:]) / price_history[-61:-1]
                returns_pct = returns * 100
                
                # 모델 설정
                p, q = self.statistical_models['garch']['order']
                model = arch.arch_model(returns_pct, vol='GARCH', p=p, q=q)
                result = model.fit(disp='off')
                
                forecast = result.forecast(horizon=forecast_days)
                vol_forecast = forecast.variance.iloc[-1].values[0]
                
                current_vol = result.conditional_volatility.iloc[-1]
                vol_change = (vol_forecast / current_vol - 1) * 100
                
                predictions['garch'] = {
                    'forecast': vol_forecast,
                    'vol_change_pct': vol_change,
                    'confidence': 0.6  # 간단한 고정 신뢰도
                }
                
                # 모델 저장
                self.statistical_models['garch']['model'] = result
            
            self.logger.info(f"통계적 모델 예측 완료: {len(predictions)}개 모델")
            
        except Exception as e:
            self.logger.warning(f"통계적 모델 실행 중 오류: {str(e)}")
        
        return predictions
    
    def integrate_statistical_predictions(self, predictions):
        """
        통계적 모델 예측 결과를 앙상블에 통합
        
        Args:
            predictions: 통계적 모델 예측 결과 딕셔너리
        
        Returns:
            float: 통합된 신호 (-1.0 ~ 1.0)
        """
        if not predictions:
            return 0.0
        
        signals = []
        weights = []
        
        # ARIMA 방향성 신호
        if 'arima' in predictions:
            signals.append(predictions['arima']['direction'])
            weights.append(self.statistical_models['arima']['weight'] * predictions['arima']['confidence'])
        
        # GARCH 변동성 예측 기반 신호
        if 'garch' in predictions:
            vol_change = predictions['garch']['vol_change_pct']
            
            # 변동성 증가 예상 시 보수적 신호(0), 감소 예상 시 중립적 신호(0.5)
            garch_signal = 0 if vol_change > 10 else (0.5 if vol_change < -10 else 0.3)
            
            signals.append(garch_signal)
            weights.append(self.statistical_models['garch']['weight'] * predictions['garch']['confidence'])
        
        # 가중 평균 계산
        if not signals:
            return 0.0
        
        return np.average(signals, weights=weights)

class EnsembleTrader:
    """
    다양한 앙상블 기법을 활용한 트레이딩 시스템
    - 모델 앙상블, 시간프레임 앙상블 통합
    - 하이브리드 접근법으로 통계적, 기술적, 강화학습 모델 통합
    """
    
    def __init__(self, config=None):
        """앙상블 트레이더 초기화"""
        self.config = config or {}
        
        # 다중 시간프레임 앙상블
        self.timeframe_ensemble = MultiTimeframeEnsemble()
        
        # 모델 앙상블
        self.model_ensemble = ModelEnsemble()
        
        # 하이브리드 접근법 사용 여부
        self.use_hybrid_approach = self.config.get('use_hybrid_approach', True)
        
        # 하이브리드 접근법 활성화 시 통계적 모델 초기화
        if self.use_hybrid_approach:
            self.model_ensemble.enable_statistical_models()
        
        # 현재 시장 상태
        self.market_regime = 'normal'
        self.regime_update_interval = self.config.get('regime_update_interval', 20)  # N회 거래마다 업데이트
        self.trade_counter = 0
        
        # 포트폴리오 및 성과 추적
        self.portfolio_value = self.config.get('initial_balance', 10000)
        self.trade_history = []
        self.performance_metrics = {}
        
        # 로깅 설정
        self.logger = logging.getLogger('EnsembleTrader')
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("앙상블 트레이더 초기화 완료")
    
    def execute_trades(self, timeframe_signals, model_predictions, price_data):
        """
        앙상블 신호를 기반으로 거래 실행
        
        Args:
            timeframe_signals: 시간프레임별 신호 {timeframe: signal}
            model_predictions: 모델별 예측 {model: prediction}
            price_data: 가격 데이터 (현재가, 이력 등)
        
        Returns:
            dict: 거래 결과
        """
        # 시장 레짐 정기 업데이트
        self.trade_counter += 1
        if self.trade_counter % self.regime_update_interval == 0 and 'history' in price_data:
            if self.use_hybrid_approach:
                self.market_regime = self.model_ensemble._detect_market_regime(price_data['history'])
                self.model_ensemble.update_regime(price_data['history'], regime=self.market_regime)
                self.logger.info(f"시장 레짐 업데이트: {self.market_regime}")
        
        # 1. 시간프레임 신호 통합
        timeframe_signal = self.timeframe_ensemble.combine_signals(timeframe_signals)
        
        # 2. 모델 예측 통합
        model_signal = self.model_ensemble.combine_predictions(model_predictions)
        
        # 3. 하이브리드 접근법 적용 (통계적 모델 통합)
        if self.use_hybrid_approach and 'history' in price_data:
            # 통계적 모델 실행
            stat_predictions = self.model_ensemble.run_statistical_models(price_data['history'])
            
            # 통계적 신호 추출
            stat_signal = self.model_ensemble.integrate_statistical_predictions(stat_predictions)
            
            # 기술적 지표 계산 및 신호 추출
            tech_signal = self._calculate_technical_signals(price_data['history'])
            
            # 시장 레짐에 따른 가중치 조정
            if self.market_regime == 'volatile':
                weights = {'timeframe': 0.2, 'model': 0.2, 'stat': 0.3, 'tech': 0.3}
            elif self.market_regime == 'trending':
                weights = {'timeframe': 0.3, 'model': 0.3, 'stat': 0.2, 'tech': 0.2}
            elif self.market_regime == 'ranging':
                weights = {'timeframe': 0.2, 'model': 0.2, 'stat': 0.2, 'tech': 0.4}
            else:  # normal
                weights = {'timeframe': 0.25, 'model': 0.25, 'stat': 0.25, 'tech': 0.25}
            
            # 최종 신호 계산 (가중 평균)
            final_signal = (
                timeframe_signal * weights['timeframe'] + 
                model_signal * weights['model'] + 
                stat_signal * weights['stat'] + 
                tech_signal * weights['tech']
            )
        else:
            # 기본 접근법 (시간프레임 + 모델 앙상블)
            final_signal = 0.5 * timeframe_signal + 0.5 * model_signal
        
        # 거래 결정
        trade_decision = self._decide_trade(final_signal)
        
        # 거래 실행
        current_price = price_data.get('current', 100.0)  # 기본값 설정
        result = self._execute_trade_decision(trade_decision, current_price)
        
        # 거래 기록 저장
        trade_record = {
            'timestamp': price_data.get('timestamp', None),
            'price': current_price,
            'action': trade_decision,
            'quantity': result.get('quantity', 0),
            'value': result.get('value', 0),
            'portfolio_value': self.portfolio_value,
            'timeframe_signal': timeframe_signal,
            'model_signal': model_signal,
            'final_signal': final_signal,
            'market_regime': self.market_regime
        }
        
        self.trade_history.append(trade_record)
        
        return {
            'action': trade_decision,
            'price': current_price,
            'result': result,
            'portfolio_value': self.portfolio_value
        }
    
    def _calculate_technical_signals(self, price_history):
        """
        기술적 지표 계산 및 신호 추출
        
        Args:
            price_history: 가격 이력 데이터
            
        Returns:
            float: 기술적 신호 (-1.0 ~ 1.0)
        """
        if len(price_history) < 30:
            return 0.0
        
        signals = []
        weights = []
        
        # 1. 이동평균 크로스오버
        short_ma = np.mean(price_history[-5:])
        long_ma = np.mean(price_history[-20:])
        
        if short_ma > long_ma * 1.01:
            ma_signal = 1.0  # 매수
        elif short_ma < long_ma * 0.99:
            ma_signal = -1.0  # 매도
        else:
            ma_signal = 0.0  # 중립
            
        signals.append(ma_signal)
        weights.append(0.3)
        
        # 2. RSI
        deltas = np.diff(price_history[-15:])
        gains = np.clip(deltas, 0, None)
        losses = -np.clip(deltas, None, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        if rsi > 70:
            rsi_signal = -1.0  # 매도 (과매수)
        elif rsi < 30:
            rsi_signal = 1.0  # 매수 (과매도)
        else:
            rsi_signal = 0.0  # 중립
            
        signals.append(rsi_signal)
        weights.append(0.3)
        
        # 3. 볼린저 밴드
        period = 20
        if len(price_history) >= period:
            sma = np.mean(price_history[-period:])
            std = np.std(price_history[-period:])
            
            upper = sma + 2 * std
            lower = sma - 2 * std
            
            current = price_history[-1]
            
            if current > upper:
                bb_signal = -1.0  # 매도 (상단돌파)
            elif current < lower:
                bb_signal = 1.0  # 매수 (하단돌파)
            else:
                # 중앙 위치 기반 신호 (중앙보다 위쪽이면 약세매도, 아래쪽이면 약세매수)
                position = (current - sma) / (upper - sma) if upper > sma else 0
                bb_signal = -position  # -1.0 ~ 1.0 사이 값
                
            signals.append(bb_signal)
            weights.append(0.2)
        
        # 4. 추세 강도
        if len(price_history) >= 10:
            returns = np.diff(price_history[-10:]) / price_history[-11:-1]
            pos_returns = sum(1 for r in returns if r > 0)
            trend_signal = (pos_returns / len(returns) - 0.5) * 2  # -1.0 ~ 1.0
            
            signals.append(trend_signal)
            weights.append(0.2)
        
        # 가중 평균 계산
        if signals:
            return np.average(signals, weights=weights)
        else:
            return 0.0
    
    def _decide_trade(self, signal):
        """
        최종 신호로부터 거래 결정
        
        Args:
            signal: 통합 신호 (-1.0 ~ 1.0)
            
        Returns:
            str: 거래 결정 ('buy', 'sell', 'hold')
        """
        # 시장 레짐에 따른 임계값 조정
        if self.market_regime == 'volatile':
            # 변동성 높은 시장에서는 보수적 임계값
            buy_threshold = 0.6
            sell_threshold = -0.6
        elif self.market_regime == 'trending':
            # 추세 시장에서는 완화된 임계값
            buy_threshold = 0.4
            sell_threshold = -0.4
        else:
            # 기본 임계값
            buy_threshold = 0.5
            sell_threshold = -0.5
        
        # 거래 결정
        if signal >= buy_threshold:
            return 'buy'
        elif signal <= sell_threshold:
            return 'sell'
        else:
            return 'hold'

# 모델 사용 예시
def test_ensemble_trader():
    # 설정
    config = {
        'timeframes': ['5m', '15m', '30m', '1h', '4h', '1d'],
        'timeframe_ensemble_method': 'weighted_average',
        'model_ensemble_method': 'weighted_average',
        'results_dir': 'ensemble_results'
    }
    
    # 앙상블 트레이더 초기화
    trader = EnsembleTrader(config)
    
    # 예시 신호 및 가격 데이터
    timeframe_signals = {
        '5m': 0.2,
        '15m': 0.3,
        '30m': 0.1,
        '1h': -0.1,
        '4h': -0.2,
        '1d': 0.4
    }
    
    model_predictions = {
        'cnn': 0.3,
        'lstm': 0.2,
        'transformer': 0.1,
        'random_forest': -0.1
    }
    
    # 가격 데이터 (간단한 예시)
    dates = pd.date_range(start='2021-01-01', periods=10, freq='D')
    prices = [100, 101, 102, 103, 102, 101, 102, 103, 104, 105]
    price_data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices)
    }, index=dates)
    
    # 거래 실행
    for i in range(len(price_data)):
        # 간단한 예시를 위해 동일한 신호 사용
        trade = trader.execute_trades(timeframe_signals, model_predictions, price_data.iloc[:i+1])
        print(f"날짜: {price_data.index[i]}, 가격: {price_data['close'].iloc[i]}, 신호: {trade['final_signal']:.2f}, 포지션: {trade['action']}")
    
    # 포트폴리오 가치 계산
    portfolio_values = trader.calculate_portfolio_value()
    
    # 성능 지표 계산
    metrics = trader.calculate_performance_metrics()
    print("\n성능 지표:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 결과 저장
    trader.save_results()
    print(f"\n결과가 {config['results_dir']}에 저장되었습니다.")

if __name__ == "__main__":
    test_ensemble_trader() 