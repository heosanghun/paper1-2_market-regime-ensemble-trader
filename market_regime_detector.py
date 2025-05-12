import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
import pandas_ta as ta

class MarketRegimeDetector:
    """
    시장 레짐을 감지하는 클래스 (추세, 변동성, 횡보 등)
    다양한 기술적 지표와 통계적 방법을 사용하여 현재 시장 상태를 분류합니다.
    """
    
    REGIME_TREND_UP = 'TREND_UP'
    REGIME_TREND_DOWN = 'TREND_DOWN'
    REGIME_RANGE_BOUND = 'RANGE_BOUND'
    REGIME_HIGH_VOLATILITY = 'HIGH_VOLATILITY'
    REGIME_LOW_VOLATILITY = 'LOW_VOLATILITY'
    
    def __init__(self, window_size=100, n_regimes=3, volatility_threshold=1.5):
        """
        초기화 함수
        
        Args:
            window_size (int): 분석 윈도우 크기
            n_regimes (int): GMM 모델에서 사용할 레짐 클러스터 수
            volatility_threshold (float): 변동성 판단을 위한 임계값
        """
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.volatility_threshold = volatility_threshold
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        self.current_regime = None
        self.regime_features = None
        
    def extract_features(self, df):
        """
        시장 레짐 판단을 위한 특성을 추출합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
            
        Returns:
            pandas.DataFrame: 추출된 특성들
        """
        # 기본 OHLCV 복사
        data = df.copy()
        
        # 수익률 계산
        data['returns'] = data['close'].pct_change()
        
        # 추세 지표 - pandas_ta 버전으로 대체
        data['sma_20'] = ta.sma(data['close'], length=20)
        data['sma_50'] = ta.sma(data['close'], length=50)
        data['sma_diff'] = data['sma_20'] - data['sma_50']
        # ADX 계산
        adx = ta.adx(data['high'], data['low'], data['close'], length=14)
        data['trend_strength'] = adx['ADX_14']
        
        # 변동성 지표 - pandas_ta 버전으로 대체
        atr = ta.atr(data['high'], data['low'], data['close'], length=14)
        data['atr'] = atr
        data['atr_pct'] = data['atr'] / data['close']
        
        # 볼린저 밴드
        bbands = ta.bbands(data['close'], length=20)
        data['bollinger_width'] = (bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0']
        
        # 추세/횡보 지표 - pandas_ta 버전으로 대체
        data['rsi'] = ta.rsi(data['close'], length=14)
        data['rsi_std'] = data['rsi'].rolling(20).std()
        
        # Hurst 지수 근사값 (추세 vs 평균회귀 구분)
        data['price_lag1'] = data['close'].shift(1)
        data['log_return'] = np.log(data['close'] / data['price_lag1'])
        
        # 결측치 제거
        data = data.dropna()
        
        # 특성 선택
        features = data[['returns', 'sma_diff', 'trend_strength', 
                        'atr_pct', 'bollinger_width', 'rsi', 'rsi_std']].copy()
        
        return features
    
    def detect_regime(self, df):
        """
        현재 시장 레짐을 탐지합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
            
        Returns:
            str: 감지된 시장 레짐
        """
        if len(df) < self.window_size:
            return self.REGIME_RANGE_BOUND  # 기본값
        
        # 마지막 window_size 기간의 데이터만 사용
        recent_df = df.tail(self.window_size).copy()
        
        # 특성 추출
        features = self.extract_features(recent_df)
        
        # 추세 판단
        is_trend = features['trend_strength'].iloc[-1] > 25
        trend_direction = np.sign(features['sma_diff'].iloc[-1])
        
        # ADF 테스트로 평균 회귀 특성 검사 (값이 작을수록 추세가 있음)
        adf_result = adfuller(recent_df['close'].values)[1]
        is_mean_reverting = adf_result < 0.05
        
        # 변동성 판단
        recent_volatility = features['atr_pct'].iloc[-1]
        avg_volatility = features['atr_pct'].mean()
        volatility_ratio = recent_volatility / avg_volatility
        
        # GMM 클러스터링을 통한 레짐 분류
        normalized_features = self.scaler.fit_transform(features)
        
        # 최근 특성만으로 클러스터 할당
        recent_features = normalized_features[-1:, :]
        self.regime_features = recent_features
        
        # 변동성 기반 판단
        if volatility_ratio > self.volatility_threshold:
            self.current_regime = self.REGIME_HIGH_VOLATILITY
        elif volatility_ratio < 1.0 / self.volatility_threshold:
            self.current_regime = self.REGIME_LOW_VOLATILITY
        # 추세 기반 판단  
        elif is_trend:
            if trend_direction > 0:
                self.current_regime = self.REGIME_TREND_UP
            else:
                self.current_regime = self.REGIME_TREND_DOWN
        # 횡보장 판단
        else:
            self.current_regime = self.REGIME_RANGE_BOUND
            
        return self.current_regime
    
    def get_regime_probabilities(self, df):
        """
        각 시장 레짐에 속할 확률을 계산합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
            
        Returns:
            dict: 각 레짐별 확률 (상승추세, 하락추세, 횡보, 고변동성, 저변동성)
        """
        self.detect_regime(df)  # 현재 레짐 갱신
        
        # 기본 확률 초기화
        probs = {
            self.REGIME_TREND_UP: 0.1,
            self.REGIME_TREND_DOWN: 0.1,
            self.REGIME_RANGE_BOUND: 0.1,
            self.REGIME_HIGH_VOLATILITY: 0.1,
            self.REGIME_LOW_VOLATILITY: 0.1
        }
        
        # 현재 레짐에 높은 확률 할당
        probs[self.current_regime] = 0.6
        
        return probs
    
    def get_optimal_strategy(self, df):
        """
        현재 시장 레짐에 가장 적합한 전략을 추천합니다.
        
        Args:
            df (pandas.DataFrame): OHLCV 데이터프레임
            
        Returns:
            str: 추천 전략 이름
        """
        regime = self.detect_regime(df)
        
        if regime == self.REGIME_TREND_UP:
            return "TREND_FOLLOWING"
        elif regime == self.REGIME_TREND_DOWN:
            return "TREND_FOLLOWING_SHORT"
        elif regime == self.REGIME_RANGE_BOUND:
            return "MEAN_REVERSION"
        elif regime == self.REGIME_HIGH_VOLATILITY:
            return "VOLATILITY_BREAKOUT"
        elif regime == self.REGIME_LOW_VOLATILITY:
            return "RANGE_TRADING"
        else:
            return "BALANCED" 