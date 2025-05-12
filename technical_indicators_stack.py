import numpy as np
import pandas as pd
import pandas_ta as ta

class TechnicalIndicatorStack:
    """
    다양한 기술적 지표(이동평균, RSI, 볼린저 밴드, MACD)를 계층화하여
    더 정확한 매매 신호를 생성하는 클래스
    """
    
    # 지표 가중치 설정 (중요도에 따라 조정 가능)
    INDICATOR_WEIGHTS = {
        'trend': 0.35,     # 추세 지표 (이동평균, MACD 등)
        'momentum': 0.25,  # 모멘텀 지표 (RSI, 스토캐스틱 등)
        'volatility': 0.20, # 변동성 지표 (볼린저 밴드, ATR 등)
        'volume': 0.20     # 거래량 지표
    }
    
    def __init__(self, use_ml_weights=False):
        """
        기술적 지표 스택 초기화
        
        Args:
            use_ml_weights (bool): 머신러닝으로 최적화된 가중치 사용 여부
        """
        self.use_ml_weights = use_ml_weights
        self.indicator_values = {}
        self.indicator_signals = {}
        
    def calculate_all_indicators(self, df):
        """
        모든 기술적 지표를 계산합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
            
        Returns:
            pandas.DataFrame: 지표가 추가된 데이터프레임
        """
        # 데이터 복사
        df_with_indicators = df.copy()
        
        # 1. 추세 지표 계산
        self._calculate_trend_indicators(df_with_indicators)
        
        # 2. 모멘텀 지표 계산
        self._calculate_momentum_indicators(df_with_indicators)
        
        # 3. 변동성 지표 계산
        self._calculate_volatility_indicators(df_with_indicators)
        
        # 4. 거래량 지표 계산
        self._calculate_volume_indicators(df_with_indicators)
        
        # 결측치 제거
        df_with_indicators = df_with_indicators.dropna()
        
        return df_with_indicators
    
    def _calculate_trend_indicators(self, df):
        """
        추세 관련 기술적 지표를 계산합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
        """
        # 이동평균
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_100'] = ta.sma(df['close'], length=100)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        # 지수 이동평균
        df['ema_10'] = ta.ema(df['close'], length=10)
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd[f'MACD_12_26_9']
        df['macd_signal'] = macd[f'MACDs_12_26_9']
        df['macd_hist'] = macd[f'MACDh_12_26_9']
        
        # ADX (추세 강도)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx[f'ADX_14']
        df['plus_di'] = adx[f'DMP_14']
        df['minus_di'] = adx[f'DMN_14']
        
        # Parabolic SAR
        psar = ta.psar(df['high'], df['low'], df['close'])
        df['sar'] = psar[f'PSARl_0.02_0.2']
        
        # Ichimoku Cloud 구성요소
        df['tenkan_sen'] = self._ichimoku_conversion_line(df['high'], df['low'], 9)
        df['kijun_sen'] = self._ichimoku_conversion_line(df['high'], df['low'], 26)
        
    def _calculate_momentum_indicators(self, df):
        """
        모멘텀 관련 기술적 지표를 계산합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
        """
        # RSI
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        
        # 스토캐스틱
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        df['slowk'] = stoch[f'STOCHk_14_3_3']
        df['slowd'] = stoch[f'STOCHd_14_3_3']
        
        # CCI (Commodity Channel Index)
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
        
        # Williams %R
        df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # Rate of Change
        df['roc'] = ta.roc(df['close'], length=10)
        
        # MFI (Money Flow Index)
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    
    def _calculate_volatility_indicators(self, df):
        """
        변동성 관련 기술적 지표를 계산합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
        """
        # 볼린저 밴드
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_upper'] = bbands[f'BBU_20_2.0']
        df['bb_middle'] = bbands[f'BBM_20_2.0']
        df['bb_lower'] = bbands[f'BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Average True Range)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = df['atr'] / df['close'] * 100  # 퍼센트 단위
        
        # Keltner Channel
        df['kc_middle'] = ta.ema(df['close'], length=20)
        df['atr_for_kc'] = ta.atr(df['high'], df['low'], df['close'], length=10)
        df['kc_upper'] = df['kc_middle'] + 2 * df['atr_for_kc']
        df['kc_lower'] = df['kc_middle'] - 2 * df['atr_for_kc']
        
    def _calculate_volume_indicators(self, df):
        """
        거래량 관련 기술적 지표를 계산합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
        """
        # OBV (On Balance Volume)
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # 거래량 이동평균
        df['volume_sma_10'] = ta.sma(df['volume'], length=10)
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
        
        # Chaikin Money Flow
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
        
        # 주가-거래량 변화율 비교
        df['price_change_pct'] = df['close'].pct_change()
        df['volume_change_pct'] = df['volume'].pct_change()
        df['pv_ratio'] = df['price_change_pct'] / df['volume_change_pct'].replace(0, 0.0001)
    
    def generate_signals(self, df, lookback=1):
        """
        각 지표별 매매 신호를 생성합니다.
        
        Args:
            df (pandas.DataFrame): 지표가 포함된 데이터프레임
            lookback (int): 신호 생성 시 참조할 과거 기간
            
        Returns:
            dict: 각 지표별 매매 신호 (-1: 매도, 0: 중립, 1: 매수)
        """
        # 신호를 담을 딕셔너리
        signals = {
            'trend': {},
            'momentum': {},
            'volatility': {},
            'volume': {}
        }
        
        # 최신 데이터 행 가져오기
        latest_data = df.iloc[-lookback:].copy()
        
        # 1. 추세 지표 신호
        # 이동평균 크로스오버
        if latest_data['sma_20'].iloc[-1] > latest_data['sma_50'].iloc[-1]:
            signals['trend']['sma_cross'] = 1  # 골든 크로스 (매수)
        elif latest_data['sma_20'].iloc[-1] < latest_data['sma_50'].iloc[-1]:
            signals['trend']['sma_cross'] = -1  # 데드 크로스 (매도)
        else:
            signals['trend']['sma_cross'] = 0
            
        # MACD 신호
        if latest_data['macd_hist'].iloc[-1] > 0:
            signals['trend']['macd'] = 1  # MACD 상승 (매수)
        elif latest_data['macd_hist'].iloc[-1] < 0:
            signals['trend']['macd'] = -1  # MACD 하락 (매도)
        else:
            signals['trend']['macd'] = 0
            
        # ADX 기반 추세 강도
        adx_value = latest_data['adx'].iloc[-1]
        plus_di = latest_data['plus_di'].iloc[-1]
        minus_di = latest_data['minus_di'].iloc[-1]
        
        if adx_value > 25:  # 강한 추세
            if plus_di > minus_di:
                signals['trend']['adx'] = 1  # 강한 상승 추세
            else:
                signals['trend']['adx'] = -1  # 강한 하락 추세
        else:
            signals['trend']['adx'] = 0  # 약한 추세 또는 횡보
            
        # 2. 모멘텀 지표 신호
        # RSI
        rsi_value = latest_data['rsi_14'].iloc[-1]
        if rsi_value > 70:
            signals['momentum']['rsi'] = -1  # 과매수 (매도 신호)
        elif rsi_value < 30:
            signals['momentum']['rsi'] = 1  # 과매도 (매수 신호)
        else:
            signals['momentum']['rsi'] = 0
            
        # 스토캐스틱
        slowk = latest_data['slowk'].iloc[-1]
        slowd = latest_data['slowd'].iloc[-1]
        
        if slowk > 80 and slowd > 80:
            signals['momentum']['stoch'] = -1  # 과매수
        elif slowk < 20 and slowd < 20:
            signals['momentum']['stoch'] = 1  # 과매도
        elif slowk > slowd:
            signals['momentum']['stoch'] = 0.5  # 약한 매수
        elif slowk < slowd:
            signals['momentum']['stoch'] = -0.5  # 약한 매도
        else:
            signals['momentum']['stoch'] = 0
            
        # 3. 변동성 지표 신호
        # 볼린저 밴드
        close = latest_data['close'].iloc[-1]
        bb_upper = latest_data['bb_upper'].iloc[-1]
        bb_lower = latest_data['bb_lower'].iloc[-1]
        
        if close > bb_upper:
            signals['volatility']['bbands'] = -1  # 저항선 돌파 (매도 신호)
        elif close < bb_lower:
            signals['volatility']['bbands'] = 1  # 지지선 돌파 (매수 신호)
        else:
            # 밴드 내부에서의 위치
            bb_position = (close - bb_lower) / (bb_upper - bb_lower)
            if bb_position > 0.8:
                signals['volatility']['bbands'] = -0.5  # 밴드 상단 접근 (약한 매도)
            elif bb_position < 0.2:
                signals['volatility']['bbands'] = 0.5  # 밴드 하단 접근 (약한 매수)
            else:
                signals['volatility']['bbands'] = 0
                
        # 4. 거래량 지표 신호
        volume = latest_data['volume'].iloc[-1]
        volume_sma_20 = latest_data['volume_sma_20'].iloc[-1]
        price_change = latest_data['price_change_pct'].iloc[-1]
        
        # 거래량 증가 + 가격 상승 = 강한 매수 신호
        if volume > volume_sma_20 * 1.5 and price_change > 0:
            signals['volume']['volume_surge'] = 1
        # 거래량 증가 + 가격 하락 = 강한 매도 신호
        elif volume > volume_sma_20 * 1.5 and price_change < 0:
            signals['volume']['volume_surge'] = -1
        else:
            signals['volume']['volume_surge'] = 0
            
        # OBV 방향
        if len(latest_data) > 1:
            obv_change = latest_data['obv'].iloc[-1] - latest_data['obv'].iloc[0]
            price_change = latest_data['close'].iloc[-1] - latest_data['close'].iloc[0]
            
            # OBV와 가격 방향이 일치하면 확인 신호, 불일치하면 반전 가능성
            if obv_change > 0 and price_change > 0:
                signals['volume']['obv'] = 1  # 확인된 상승
            elif obv_change < 0 and price_change < 0:
                signals['volume']['obv'] = -1  # 확인된 하락
            elif obv_change > 0 and price_change < 0:
                signals['volume']['obv'] = 0.5  # 반전 가능성 (매수)
            elif obv_change < 0 and price_change > 0:
                signals['volume']['obv'] = -0.5  # 반전 가능성 (매도)
            else:
                signals['volume']['obv'] = 0
                
        self.indicator_signals = signals
        return signals
    
    def get_aggregated_signal(self, signals=None):
        """
        모든 지표의 신호를 통합하여 최종 매매 신호를 생성합니다.
        
        Args:
            signals (dict, optional): 각 지표별 신호, 없으면 저장된 신호 사용
            
        Returns:
            dict: 통합된 매매 신호와 신뢰도
        """
        if signals is None:
            signals = self.indicator_signals
            
        if not signals:
            return {'signal': 0, 'confidence': 0, 'signal_type': 'NEUTRAL'}
            
        # 카테고리별 신호 평균
        category_signals = {}
        
        for category, indicators in signals.items():
            if indicators:
                category_signals[category] = sum(indicators.values()) / len(indicators)
            else:
                category_signals[category] = 0
                
        # 가중 평균으로 최종 신호 계산
        final_signal = 0
        total_weight = 0
        
        for category, signal in category_signals.items():
            weight = self.INDICATOR_WEIGHTS.get(category, 0.25)  # 기본 가중치는 0.25
            final_signal += signal * weight
            total_weight += weight
            
        if total_weight > 0:
            final_signal = final_signal / total_weight
            
        # 신호 신뢰도 계산 (-1 ~ 1 범위에서 0에 가까울수록 신뢰도 낮음)
        confidence = abs(final_signal)
        
        # 신호 분류
        signal_type = 'NEUTRAL'
        if final_signal > 0.7:
            signal_type = 'STRONG_BUY'
        elif final_signal > 0.3:
            signal_type = 'BUY'
        elif final_signal < -0.7:
            signal_type = 'STRONG_SELL'
        elif final_signal < -0.3:
            signal_type = 'SELL'
            
        return {
            'signal': final_signal,  # 원시 신호 값 (-1 ~ 1)
            'confidence': confidence,  # 신호 신뢰도 (0 ~ 1)
            'signal_type': signal_type,  # 신호 유형
            'category_signals': category_signals  # 카테고리별 신호
        }
    
    def _ichimoku_conversion_line(self, high, low, period):
        """
        Ichimoku 전환선 계산
        
        Args:
            high (pandas.Series): 고가 시리즈
            low (pandas.Series): 저가 시리즈
            period (int): 기간
            
        Returns:
            pandas.Series: 전환선 값
        """
        period_high = high.rolling(window=period).max()
        period_low = low.rolling(window=period).min()
        return (period_high + period_low) / 2 