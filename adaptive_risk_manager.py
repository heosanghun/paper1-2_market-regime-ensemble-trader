import numpy as np
import pandas as pd

class AdaptiveRiskManager:
    """
    시장 상황에 따라 포지션 크기를 조절하고 연속 손실 시 보수적 접근으로 전환하는 리스크 관리 시스템
    """
    
    # 리스크 레벨 상수
    RISK_LEVEL_HIGH = 'HIGH'  # 공격적 접근
    RISK_LEVEL_MEDIUM = 'MEDIUM'  # 표준 접근
    RISK_LEVEL_LOW = 'LOW'  # 보수적 접근
    RISK_LEVEL_MINIMAL = 'MINIMAL'  # 최소 리스크 (회복 모드)
    
    def __init__(self, 
                 initial_capital=10000, 
                 max_position_size=0.2,  # 최대 포지션 크기 (자본의 %)
                 drawdown_threshold=0.1,  # 최대 허용 드로다운
                 consecutive_loss_threshold=3,  # 연속 손실 기준값
                 volatility_scaling=True,  # 변동성 기반 포지션 크기 조정 여부
                 kelly_fraction=0.5):  # Kelly 기준 계수 (보수적 접근)
        """
        리스크 관리자 초기화
        
        Args:
            initial_capital (float): 초기 자본금
            max_position_size (float): 최대 포지션 크기 (자본의 %)
            drawdown_threshold (float): 최대 허용 드로다운 (자본의 %)
            consecutive_loss_threshold (int): 연속 손실 허용 횟수
            volatility_scaling (bool): 변동성 기반 포지션 크기 조정 여부
            kelly_fraction (float): Kelly 기준 계수 (1보다 작을수록 보수적)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.drawdown_threshold = drawdown_threshold
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.volatility_scaling = volatility_scaling
        self.kelly_fraction = kelly_fraction
        
        # 상태 변수 초기화
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.trade_history = []
        self.current_risk_level = self.RISK_LEVEL_MEDIUM
        self.position_sizing_history = []
        
    def update_capital(self, pnl):
        """
        거래 결과를 바탕으로 자본금과 성과 지표를 업데이트
        
        Args:
            pnl (float): 손익 금액
            
        Returns:
            float: 업데이트된 현재 자본금
        """
        old_capital = self.current_capital
        self.current_capital += pnl
        
        # 손익 기록 업데이트
        trade_result = {
            'pnl': pnl,
            'pnl_pct': pnl / old_capital,
            'capital_before': old_capital,
            'capital_after': self.current_capital,
            'risk_level': self.current_risk_level
        }
        self.trade_history.append(trade_result)
        
        # 피크 자본 업데이트
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            
        # 현재 드로다운 계산
        self.current_drawdown = 1 - (self.current_capital / self.peak_capital)
        
        # 연속 손익 업데이트
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # 리스크 레벨 업데이트
        self._update_risk_level()
        
        return self.current_capital
    
    def _update_risk_level(self):
        """
        현재 성과와 시장 상황에 따라 리스크 레벨을 업데이트
        """
        # 연속 손실에 따른 리스크 감소
        if self.consecutive_losses >= self.consecutive_loss_threshold * 2:
            self.current_risk_level = self.RISK_LEVEL_MINIMAL
        elif self.consecutive_losses >= self.consecutive_loss_threshold:
            self.current_risk_level = self.RISK_LEVEL_LOW
        # 큰 드로다운에 따른 리스크 감소
        elif self.current_drawdown > self.drawdown_threshold:
            self.current_risk_level = self.RISK_LEVEL_LOW
        # 연속 이익에 따른 리스크 증가 (선택적)
        elif self.consecutive_wins >= 3 and self.current_drawdown < self.drawdown_threshold / 2:
            self.current_risk_level = self.RISK_LEVEL_HIGH
        # 기본 리스크 레벨
        else:
            self.current_risk_level = self.RISK_LEVEL_MEDIUM
    
    def get_position_size(self, expected_win_rate=0.5, expected_gain_loss_ratio=1.0, 
                          market_volatility=None, confidence=None):
        """
        현재 리스크 레벨과 시장 상황에 따라 적절한 포지션 크기를 계산
        
        Args:
            expected_win_rate (float): 예상 승률 (0~1)
            expected_gain_loss_ratio (float): 예상 수익:손실 비율
            market_volatility (float, optional): 시장 변동성 (표준화된 값)
            confidence (float, optional): 거래 신호 신뢰도 (0~1)
            
        Returns:
            float: 자본 대비 포지션 크기 비율 (0~max_position_size)
        """
        # 켈리 기준 포지션 크기 계산
        kelly = expected_win_rate - ((1 - expected_win_rate) / expected_gain_loss_ratio)
        
        # 보수적 조정 (켈리 기준의 일부만 사용)
        kelly = max(0, kelly * self.kelly_fraction)
        
        # 포지션 크기 초기화
        position_size = min(kelly, self.max_position_size)
        
        # 리스크 레벨 기반 조정
        if self.current_risk_level == self.RISK_LEVEL_HIGH:
            position_size = min(position_size * 1.2, self.max_position_size)
        elif self.current_risk_level == self.RISK_LEVEL_LOW:
            position_size = position_size * 0.5
        elif self.current_risk_level == self.RISK_LEVEL_MINIMAL:
            position_size = position_size * 0.25
        
        # 변동성 기반 조정 (선택적)
        if self.volatility_scaling and market_volatility is not None:
            # 변동성이 높을수록 포지션 크기 감소
            volatility_factor = 1.0 / (1.0 + market_volatility)
            position_size = position_size * volatility_factor
        
        # 신호 신뢰도 기반 조정 (선택적)
        if confidence is not None:
            position_size = position_size * confidence
        
        # 최종 포지션 크기 기록
        self.position_sizing_history.append({
            'capital': self.current_capital,
            'position_size': position_size,
            'risk_level': self.current_risk_level,
            'consecutive_losses': self.consecutive_losses,
            'drawdown': self.current_drawdown
        })
        
        return position_size
    
    def get_risk_adjusted_amount(self, expected_win_rate=0.5, expected_gain_loss_ratio=1.0, 
                                market_volatility=None, confidence=None):
        """
        투자 금액을 계산합니다.
        
        Args:
            expected_win_rate (float): 예상 승률 (0~1)
            expected_gain_loss_ratio (float): 예상 수익:손실 비율
            market_volatility (float, optional): 시장 변동성 (표준화된 값)
            confidence (float, optional): 거래 신호 신뢰도 (0~1)
            
        Returns:
            float: 투자 금액
        """
        position_size_pct = self.get_position_size(
            expected_win_rate, expected_gain_loss_ratio, market_volatility, confidence
        )
        return self.current_capital * position_size_pct
    
    def calculate_stop_loss(self, entry_price, position_type='long', 
                           atr_value=None, fixed_pct=0.02):
        """
        적절한 손절 가격을 계산합니다.
        
        Args:
            entry_price (float): 진입 가격
            position_type (str): 포지션 유형 ('long' 또는 'short')
            atr_value (float, optional): ATR 값 (있을 경우 ATR 기반 손절 계산)
            fixed_pct (float): ATR 값이 없을 경우 사용할 고정 손절 비율
            
        Returns:
            float: 손절 가격
        """
        # 리스크 레벨에 따른 ATR 배수 조정
        if self.current_risk_level == self.RISK_LEVEL_HIGH:
            atr_multiplier = 2.0
        elif self.current_risk_level == self.RISK_LEVEL_MEDIUM:
            atr_multiplier = 1.5
        elif self.current_risk_level == self.RISK_LEVEL_LOW:
            atr_multiplier = 1.0
        else:  # MINIMAL
            atr_multiplier = 0.7
        
        # ATR 기반 손절폭 계산
        if atr_value is not None:
            stop_distance = atr_value * atr_multiplier
        else:
            # 고정 비율 손절
            stop_distance = entry_price * fixed_pct
        
        # 롱/숏에 따른 손절가 계산
        if position_type.lower() == 'long':
            stop_price = entry_price - stop_distance
        else:  # short
            stop_price = entry_price + stop_distance
            
        return stop_price
    
    def adjust_position_for_market_regime(self, market_regime, base_position_size):
        """
        시장 레짐에 따라 포지션 크기를 조정합니다.
        
        Args:
            market_regime (str): 현재 시장 레짐 (예: 'TREND_UP', 'RANGE_BOUND' 등)
            base_position_size (float): 기본 포지션 크기
            
        Returns:
            float: 조정된 포지션 크기
        """
        # 시장 레짐별 조정 계수
        regime_adjustments = {
            'TREND_UP': 1.2,      # 상승 추세에서 포지션 증가
            'TREND_DOWN': 1.1,    # 하락 추세에서도 (숏 포지션) 약간 증가
            'RANGE_BOUND': 0.8,   # 횡보 시장에서 포지션 감소
            'HIGH_VOLATILITY': 0.6,  # 고변동성 시장에서 큰 폭 감소
            'LOW_VOLATILITY': 1.0    # 저변동성 시장에서 정상 유지
        }
        
        # 기본값 설정
        adjustment_factor = regime_adjustments.get(market_regime, 1.0)
        
        # 리스크 레벨이 낮을 경우 추가 축소
        if self.current_risk_level in [self.RISK_LEVEL_LOW, self.RISK_LEVEL_MINIMAL]:
            adjustment_factor *= 0.8
            
        # 조정된 포지션 크기 (최대 한도 제한)
        adjusted_size = min(base_position_size * adjustment_factor, self.max_position_size)
        
        return adjusted_size
    
    def get_performance_metrics(self):
        """
        현재까지의 성과 지표를 계산합니다.
        
        Returns:
            dict: 주요 성과 지표
        """
        if not self.trade_history:
            return {
                'total_return': 0.0,
                'drawdown': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'risk_level': self.current_risk_level
            }
            
        # 데이터 프레임 변환
        trade_df = pd.DataFrame(self.trade_history)
        
        # 총 수익률
        total_return = (self.current_capital / self.initial_capital) - 1.0
        
        # 승률 계산
        if len(trade_df) > 0:
            wins = trade_df[trade_df['pnl'] > 0]
            losses = trade_df[trade_df['pnl'] < 0]
            win_rate = len(wins) / len(trade_df) if len(trade_df) > 0 else 0.0
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0.0
            avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
        
        return {
            'total_return': total_return,
            'drawdown': self.current_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_level': self.current_risk_level,
            'consecutive_losses': self.consecutive_losses
        } 