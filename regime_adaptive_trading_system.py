import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

from market_regime_detector import MarketRegimeDetector
from multi_timeframe_analyzer import MultiTimeframeAnalyzer
from statistical_models import StatisticalModelIntegration
from adaptive_risk_manager import AdaptiveRiskManager
from technical_indicators_stack import TechnicalIndicatorStack

class RegimeAdaptiveTradingSystem:
    """
    시장 레짐에 적응하는 트레이딩 시스템
    1. 시장 레짐 감지
    2. 다중 타임프레임 분석
    3. 통계적 모델 통합
    4. 적응적 리스크 관리
    5. 기술적 지표 계층화
    """
    
    def __init__(self, initial_capital=10000, max_position_size=0.2):
        """
        시장 레짐 적응형 트레이딩 시스템 초기화
        
        Args:
            initial_capital (float): 초기 자본금
            max_position_size (float): 최대 포지션 크기 (자본의 %)
        """
        # 구성 요소 초기화
        self.regime_detector = MarketRegimeDetector()
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer()
        self.statistical_models = StatisticalModelIntegration()
        self.risk_manager = AdaptiveRiskManager(
            initial_capital=initial_capital,
            max_position_size=max_position_size
        )
        self.technical_indicators = TechnicalIndicatorStack()
        
        # 시스템 상태
        self.current_regime = None
        self.current_position = None  # 'long', 'short', None
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # 성과 지표
        self.trades_history = []
        self.regime_history = []
        self.signals_history = []
        
        # 로깅 설정
        self.log_enabled = True
        self.log_file = os.path.join("results", "regime_adaptive_system.log")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
    def load_data(self, timeframe_data_dict):
        """
        여러 타임프레임의 데이터를 로드합니다.
        
        Args:
            timeframe_data_dict (dict): 타임프레임별 OHLCV 데이터프레임
                예: {'1h': df_1h, '4h': df_4h, '1d': df_1d}
        """
        # 데이터 검증
        for tf, df in timeframe_data_dict.items():
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"데이터프레임 {tf}에 필요한 열이 없습니다: {required_columns}")
                
        # 다중 타임프레임 분석기에 데이터 추가
        for tf, df in timeframe_data_dict.items():
            self.multi_timeframe_analyzer.add_timeframe_data(tf, df)
            
        # 가장 짧은 타임프레임을 주요 데이터로 설정
        shortest_tf = sorted(timeframe_data_dict.keys(), 
                           key=lambda x: {'1m': 1, '5m': 2, '15m': 3, '1h': 4, '4h': 5, '1d': 6}[x])[0]
        self.primary_data = timeframe_data_dict[shortest_tf]
        
        # 기술적 지표 계산
        self.primary_data_with_indicators = self.technical_indicators.calculate_all_indicators(self.primary_data)
        
        # 통계적 모델 학습
        self.statistical_models.fit_arima(self.primary_data['close'])
        self.statistical_models.fit_garch(self.primary_data['close'])
        
        # 초기 시장 레짐 감지
        self.current_regime = self.regime_detector.detect_regime(self.primary_data)
        self.log(f"초기 시장 레짐: {self.current_regime}")
        
    def analyze_market(self):
        """
        현재 시장 상황을 분석하고 통합된 신호를 생성합니다.
        
        Returns:
            dict: 통합된 시장 분석 결과
        """
        # 1. 시장 레짐 감지
        self.current_regime = self.regime_detector.detect_regime(self.primary_data)
        regime_probs = self.regime_detector.get_regime_probabilities(self.primary_data)
        
        # 2. 다중 타임프레임 신호 생성
        mtf_signals = self.multi_timeframe_analyzer.generate_multi_timeframe_signals()
        
        # 3. 기술적 지표 기반 신호 생성
        tech_signals = self.technical_indicators.generate_signals(self.primary_data_with_indicators)
        aggregated_tech_signal = self.technical_indicators.get_aggregated_signal(tech_signals)
        
        # 4. 통계적 모델 예측
        current_price = self.primary_data['close'].iloc[-1]
        forecast_features = self.statistical_models.generate_combined_features(current_price)
        confidence_interval = self.statistical_models.calculate_confidence_interval(current_price)
        trend_probability = self.statistical_models.get_trend_probability()
        future_regime = self.statistical_models.get_regime_prediction()
        
        # 각 분석 결과 통합
        analysis_result = {
            'current_regime': self.current_regime,
            'regime_probabilities': regime_probs,
            'multi_timeframe_signal': mtf_signals,
            'technical_signal': aggregated_tech_signal,
            'statistical_forecast': {
                'trend_probability': trend_probability,
                'confidence_interval': confidence_interval,
                'future_regime': future_regime
            },
            'current_price': current_price,
            'timestamp': self.primary_data.index[-1]
        }
        
        # 결과 기록
        self.regime_history.append({
            'timestamp': self.primary_data.index[-1],
            'regime': self.current_regime,
            'regime_probs': regime_probs
        })
        
        self.signals_history.append({
            'timestamp': self.primary_data.index[-1],
            'mtf_signal': mtf_signals.get('signal', 'NEUTRAL'),
            'tech_signal': aggregated_tech_signal.get('signal_type', 'NEUTRAL'),
            'trend_prob': trend_probability,
            'future_regime': future_regime
        })
        
        return analysis_result
    
    def generate_trading_decision(self, analysis_result):
        """
        시장 분석 결과를 바탕으로 거래 결정을 생성합니다.
        
        Args:
            analysis_result (dict): 시장 분석 결과
            
        Returns:
            dict: 거래 결정 (action, position_size, stop_loss, take_profit)
        """
        # 기본 결정 초기화
        decision = {
            'action': 'HOLD',  # 'BUY', 'SELL', 'HOLD', 'CLOSE'
            'position_size': 0,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0
        }
        
        # 각 신호 소스별 매매 방향 추출
        mtf_signal = analysis_result['multi_timeframe_signal'].get('signal', 'NEUTRAL')
        tech_signal = analysis_result['technical_signal'].get('signal_type', 'NEUTRAL')
        trend_prob = analysis_result['statistical_forecast']['trend_probability']
        stat_signal = 'BUY' if trend_prob > 0.6 else ('SELL' if trend_prob < 0.4 else 'NEUTRAL')
        
        # 현재 레짐
        current_regime = analysis_result['current_regime']
        
        # 레짐별 최적 전략 가져오기
        optimal_strategy = self.regime_detector.get_optimal_strategy(self.primary_data)
        
        # 레짐별 거래 결정 로직
        if optimal_strategy == "TREND_FOLLOWING":
            # 추세 추종 전략 - 기술적 신호와 다중 타임프레임 신호에 높은 가중치
            tech_weight, mtf_weight, stat_weight = 0.4, 0.4, 0.2
        elif optimal_strategy == "MEAN_REVERSION":
            # 평균 회귀 전략 - 통계적 예측과 과매수/과매도 지표에 더 높은 가중치
            tech_weight, mtf_weight, stat_weight = 0.3, 0.3, 0.4
        elif optimal_strategy == "VOLATILITY_BREAKOUT":
            # 변동성 돌파 전략 - 기술적 신호와 다중 타임프레임 신호에 집중
            tech_weight, mtf_weight, stat_weight = 0.5, 0.4, 0.1
        elif optimal_strategy == "RANGE_TRADING":
            # 레인지 거래 전략 - 기술적 지표와 통계적 예측에 집중
            tech_weight, mtf_weight, stat_weight = 0.45, 0.1, 0.45
        else:  # 균형 전략
            tech_weight, mtf_weight, stat_weight = 0.33, 0.33, 0.34
            
        # 신호 통합
        buy_confidence = 0
        sell_confidence = 0
        
        # 기술적 신호 반영
        if tech_signal in ['STRONG_BUY', 'BUY']:
            buy_confidence += tech_weight * (1.0 if tech_signal == 'STRONG_BUY' else 0.7)
        elif tech_signal in ['STRONG_SELL', 'SELL']:
            sell_confidence += tech_weight * (1.0 if tech_signal == 'STRONG_SELL' else 0.7)
            
        # 다중 타임프레임 신호 반영
        if mtf_signal in ['STRONG_BUY', 'BUY']:
            buy_confidence += mtf_weight * (1.0 if mtf_signal == 'STRONG_BUY' else 0.7)
        elif mtf_signal in ['STRONG_SELL', 'SELL']:
            sell_confidence += mtf_weight * (1.0 if mtf_signal == 'STRONG_SELL' else 0.7)
            
        # 통계적 신호 반영
        if stat_signal == 'BUY':
            buy_confidence += stat_weight * trend_prob
        elif stat_signal == 'SELL':
            sell_confidence += stat_weight * (1 - trend_prob)
            
        # 최종 결정
        confidence = max(buy_confidence, sell_confidence)
        
        # 현재 가격
        current_price = analysis_result['current_price']
        
        # 기존 포지션이 있는 경우의 처리
        if self.current_position == 'long':
            # 매도 신호가 강하면 청산
            if sell_confidence > 0.6:
                decision['action'] = 'CLOSE'
                decision['confidence'] = sell_confidence
            else:
                decision['action'] = 'HOLD'
        elif self.current_position == 'short':
            # 매수 신호가 강하면 청산
            if buy_confidence > 0.6:
                decision['action'] = 'CLOSE'
                decision['confidence'] = buy_confidence
            else:
                decision['action'] = 'HOLD'
        else:  # 포지션 없음
            if buy_confidence > 0.5 and buy_confidence > sell_confidence:
                decision['action'] = 'BUY'
                decision['confidence'] = buy_confidence
            elif sell_confidence > 0.5 and sell_confidence > buy_confidence:
                decision['action'] = 'SELL'
                decision['confidence'] = sell_confidence
                
        # 포지션 크기 결정 (만약 BUY/SELL 액션인 경우)
        if decision['action'] in ['BUY', 'SELL']:
            # 예상 승률 계산 (신뢰도 기반)
            expected_win_rate = min(0.75, 0.5 + decision['confidence'] * 0.5)
            
            # 예상 수익:손실 비율 계산
            if decision['action'] == 'BUY':
                # 통계적 예측에서 상단 신뢰구간 사용
                upper_bound = analysis_result['statistical_forecast']['confidence_interval'][1]
                expected_gain_ratio = (upper_bound - current_price) / current_price
            else:  # SELL
                # 통계적 예측에서 하단 신뢰구간 사용
                lower_bound = analysis_result['statistical_forecast']['confidence_interval'][0]
                expected_gain_ratio = (current_price - lower_bound) / current_price
                
            expected_gain_ratio = max(0.01, expected_gain_ratio)  # 최소 1%
            expected_loss_ratio = 0.01  # 기본 손실 비율 (나중에 ATR로 조정)
            expected_gain_loss_ratio = expected_gain_ratio / expected_loss_ratio
            
            # 시장 변동성 계산 (ATR 기반)
            market_volatility = self.primary_data_with_indicators['atr_pct'].iloc[-1] / 1.5
            
            # 포지션 크기 계산
            base_position_size = self.risk_manager.get_position_size(
                expected_win_rate=expected_win_rate,
                expected_gain_loss_ratio=expected_gain_loss_ratio,
                market_volatility=market_volatility,
                confidence=decision['confidence']
            )
            
            # 시장 레짐에 따른 포지션 크기 조정
            position_size = self.risk_manager.adjust_position_for_market_regime(
                current_regime, base_position_size
            )
            
            decision['position_size'] = position_size
            
            # 손절가 계산
            if 'atr' in self.primary_data_with_indicators.columns:
                atr_value = self.primary_data_with_indicators['atr'].iloc[-1]
                position_type = 'long' if decision['action'] == 'BUY' else 'short'
                decision['stop_loss'] = self.risk_manager.calculate_stop_loss(
                    current_price, position_type, atr_value
                )
            else:
                # ATR 없을 경우 고정 비율 손절
                stop_pct = 0.02  # 2%
                if decision['action'] == 'BUY':
                    decision['stop_loss'] = current_price * (1 - stop_pct)
                else:  # SELL
                    decision['stop_loss'] = current_price * (1 + stop_pct)
                    
            # 이익실현가 계산 (손절폭의 1.5배)
            if decision['action'] == 'BUY':
                stop_distance = current_price - decision['stop_loss']
                decision['take_profit'] = current_price + (stop_distance * 1.5)
            else:  # SELL
                stop_distance = decision['stop_loss'] - current_price
                decision['take_profit'] = current_price - (stop_distance * 1.5)
                
        return decision
    
    def execute_trade(self, decision, current_price):
        """
        트레이딩 결정을 실행합니다.
        
        Args:
            decision (dict): 거래 결정
            current_price (float): 현재 가격
            
        Returns:
            dict: 거래 실행 결과
        """
        action = decision['action']
        result = {
            'timestamp': datetime.now(),
            'action': action,
            'price': current_price,
            'position_size': 0,
            'pnl': 0
        }
        
        # 액션에 따른 처리
        if action == 'BUY' and self.current_position is None:
            # 신규 롱 포지션 진입
            self.current_position = 'long'
            self.position_size = decision['position_size']
            self.entry_price = current_price
            self.stop_loss = decision['stop_loss']
            self.take_profit = decision['take_profit']
            
            result['position_size'] = self.position_size
            result['entry_price'] = self.entry_price
            result['stop_loss'] = self.stop_loss
            result['take_profit'] = self.take_profit
            
            self.log(f"매수 진입: 가격={current_price}, 크기={self.position_size}, SL={self.stop_loss}, TP={self.take_profit}")
            
        elif action == 'SELL' and self.current_position is None:
            # 신규 숏 포지션 진입
            self.current_position = 'short'
            self.position_size = decision['position_size']
            self.entry_price = current_price
            self.stop_loss = decision['stop_loss']
            self.take_profit = decision['take_profit']
            
            result['position_size'] = self.position_size
            result['entry_price'] = self.entry_price
            result['stop_loss'] = self.stop_loss
            result['take_profit'] = self.take_profit
            
            self.log(f"매도 진입: 가격={current_price}, 크기={self.position_size}, SL={self.stop_loss}, TP={self.take_profit}")
            
        elif action == 'CLOSE' and self.current_position is not None:
            # 포지션 청산
            position_size = self.position_size
            entry_price = self.entry_price
            position_type = self.current_position
            
            # 손익 계산
            if position_type == 'long':
                pnl = (current_price - entry_price) / entry_price * position_size
            else:  # short
                pnl = (entry_price - current_price) / entry_price * position_size
                
            # 자본금 업데이트
            capital_change = pnl * self.risk_manager.current_capital
            self.risk_manager.update_capital(capital_change)
            
            result['position_size'] = position_size
            result['entry_price'] = entry_price
            result['exit_price'] = current_price
            result['pnl'] = pnl
            result['pnl_amount'] = capital_change
            
            self.log(f"포지션 청산: 타입={position_type}, 진입={entry_price}, 청산={current_price}, PnL={pnl:.4f}")
            
            # 포지션 초기화
            self.current_position = None
            self.position_size = 0
            self.entry_price = 0
            self.stop_loss = 0
            self.take_profit = 0
            
        # 거래 기록 추가
        self.trades_history.append(result)
        
        return result
    
    def check_stop_levels(self, current_price):
        """
        현재 가격이 손절 또는 이익실현 레벨에 도달했는지 확인합니다.
        
        Args:
            current_price (float): 현재 가격
            
        Returns:
            str or None: 트리거된 레벨 ('stop_loss', 'take_profit', None)
        """
        if self.current_position is None:
            return None
            
        if self.current_position == 'long':
            if current_price <= self.stop_loss:
                return 'stop_loss'
            if current_price >= self.take_profit:
                return 'take_profit'
        else:  # short
            if current_price >= self.stop_loss:
                return 'stop_loss'
            if current_price <= self.take_profit:
                return 'take_profit'
                
        return None
    
    def update_stop_levels(self, current_price):
        """
        트레일링 스탑 등 스탑 레벨을 업데이트합니다.
        
        Args:
            current_price (float): 현재 가격
        """
        if self.current_position is None:
            return
            
        # 트레일링 스탑 로직
        if self.current_position == 'long':
            # 현재 가격이 진입가보다 2% 이상 상승했을 때 손절가를 진입가로 이동
            if current_price > self.entry_price * 1.02 and self.stop_loss < self.entry_price:
                self.stop_loss = self.entry_price
                self.log(f"트레일링 스탑 업데이트 (롱): SL={self.stop_loss}")
                
            # 현재 가격이 진입가보다 5% 이상 상승했을 때 손절가를 진입가의 1.5%로 이동
            elif current_price > self.entry_price * 1.05 and self.stop_loss < self.entry_price * 1.015:
                self.stop_loss = self.entry_price * 1.015
                self.log(f"트레일링 스탑 업데이트 (롱): SL={self.stop_loss}")
                
        else:  # short
            # 현재 가격이 진입가보다 2% 이상 하락했을 때 손절가를 진입가로 이동
            if current_price < self.entry_price * 0.98 and self.stop_loss > self.entry_price:
                self.stop_loss = self.entry_price
                self.log(f"트레일링 스탑 업데이트 (숏): SL={self.stop_loss}")
                
            # 현재 가격이 진입가보다 5% 이상 하락했을 때 손절가를 진입가의 1.5% 아래로 이동
            elif current_price < self.entry_price * 0.95 and self.stop_loss > self.entry_price * 0.985:
                self.stop_loss = self.entry_price * 0.985
                self.log(f"트레일링 스탑 업데이트 (숏): SL={self.stop_loss}")
    
    def process_bar(self, new_bar_data):
        """
        새로운 가격 데이터가 도착할 때 전체 트레이딩 프로세스 실행
        
        Args:
            new_bar_data (pandas.DataFrame): 새로운 OHLCV 데이터
            
        Returns:
            dict: 처리 결과
        """
        # 데이터 업데이트
        self.primary_data = pd.concat([self.primary_data, new_bar_data]).tail(1000)
        
        # 지표 업데이트
        self.primary_data_with_indicators = self.technical_indicators.calculate_all_indicators(self.primary_data)
        
        # 현재 가격
        current_price = new_bar_data['close'].iloc[-1]
        
        # 스탑 레벨 체크
        triggered_level = self.check_stop_levels(current_price)
        if triggered_level:
            # 스탑 레벨 트리거시 포지션 청산
            self.log(f"{triggered_level.upper()} 트리거: 가격={current_price}")
            close_decision = {'action': 'CLOSE'}
            self.execute_trade(close_decision, current_price)
            return {'action': 'CLOSE_BY_STOP', 'triggered_level': triggered_level}
        
        # 스탑 레벨 업데이트 (트레일링 스탑)
        self.update_stop_levels(current_price)
        
        # 시장 분석
        analysis_result = self.analyze_market()
        
        # 거래 결정 생성
        decision = self.generate_trading_decision(analysis_result)
        
        # 거래 실행
        if decision['action'] != 'HOLD':
            execution_result = self.execute_trade(decision, current_price)
            decision.update(execution_result)
            
        return decision
        
    def get_performance_metrics(self):
        """
        트레이딩 시스템의 성과 지표를 계산합니다.
        
        Returns:
            dict: 성과 지표
        """
        # 리스크 매니저에서 성과 지표 가져오기
        risk_metrics = self.risk_manager.get_performance_metrics()
        
        # 거래 성과 계산
        trades_df = pd.DataFrame(self.trades_history)
        
        # 추가 지표 계산
        if not trades_df.empty and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            total_trades = len(trades_df)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
            
            # 평균 수익:손실 비율
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            
            # 최대 연속 손실
            pnl_series = trades_df['pnl'].values
            max_consecutive_losses = 0
            current_consecutive_losses = 0
            
            for pnl in pnl_series:
                if pnl < 0:
                    current_consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
                else:
                    current_consecutive_losses = 0
        else:
            total_trades = 0
            win_rate = 0
            win_loss_ratio = 0
            max_consecutive_losses = 0
            
        # 레짐 분석
        regime_counts = {}
        if self.regime_history:
            for record in self.regime_history:
                regime = record['regime']
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
        # 최종 성과 지표
        performance = {
            'total_return': risk_metrics['total_return'],
            'drawdown': risk_metrics['drawdown'],
            'total_trades': total_trades,
            'win_rate': win_rate,
            'win_loss_ratio': win_loss_ratio,
            'max_consecutive_losses': max_consecutive_losses,
            'current_risk_level': risk_metrics['risk_level'],
            'regime_distribution': regime_counts
        }
        
        return performance
        
    def log(self, message):
        """
        로그 메시지를 기록합니다.
        
        Args:
            message (str): 로그 메시지
        """
        if not self.log_enabled:
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        # 파일에 로그 기록
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n') 