import pandas as pd
import numpy as np
import pmdarima as pm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings('ignore')

class StatisticalModelIntegration:
    """
    ARIMA, GARCH 같은 통계적 모델을 활용하여 시계열 예측을 수행하고
    강화학습 모델과 통합하기 위한 클래스
    """
    
    def __init__(self, n_forecast=5):
        """
        초기화 함수
        
        Args:
            n_forecast (int): 예측할 미래 시점 수
        """
        self.n_forecast = n_forecast
        self.arima_model = None
        self.garch_model = None
        self.last_arima_forecast = None
        self.last_volatility_forecast = None
        self.arima_order = None
        self.garch_order = (1, 1)  # 기본 GARCH(1,1) 모델
        
    def fit_arima(self, prices, auto_order=True):
        """
        ARIMA 모델을 학습합니다.
        
        Args:
            prices (pandas.Series): 가격 시계열 데이터
            auto_order (bool): 자동으로 최적의 ARIMA 차수를 찾을지 여부
            
        Returns:
            bool: 모델 학습 성공 여부
        """
        # 데이터 준비
        log_prices = np.log(prices)
        
        try:
            if auto_order:
                # 자동으로 최적의 ARIMA 모델 찾기
                self.arima_model = pm.auto_arima(
                    log_prices,
                    start_p=0, start_q=0,
                    max_p=4, max_q=4, max_d=2,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
                self.arima_order = self.arima_model.order
            else:
                # 기본 ARIMA(1,1,1) 모델 사용
                self.arima_order = (1, 1, 1)
                self.arima_model = ARIMA(log_prices, order=self.arima_order)
                self.arima_model = self.arima_model.fit()
                
            return True
        except Exception as e:
            print(f"ARIMA 모델 학습 실패: {e}")
            # 간단한 백업 모델 사용
            self.arima_order = (1, 1, 0)
            try:
                self.arima_model = ARIMA(log_prices, order=self.arima_order)
                self.arima_model = self.arima_model.fit()
                return True
            except:
                return False
    
    def fit_garch(self, prices):
        """
        GARCH 모델을 학습하여 변동성을 예측합니다.
        
        Args:
            prices (pandas.Series): 가격 시계열 데이터
            
        Returns:
            bool: 모델 학습 성공 여부
        """
        # 로그 수익률 계산
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        try:
            # GARCH(1,1) 모델 학습
            self.garch_model = arch_model(
                log_returns * 100,  # 스케일링
                vol='Garch', 
                p=self.garch_order[0], 
                q=self.garch_order[1],
                mean='Zero',  # 평균이 0이라고 가정
                rescale=False
            )
            self.garch_fit = self.garch_model.fit(disp='off')
            return True
        except Exception as e:
            print(f"GARCH 모델 학습 실패: {e}")
            return False
    
    def predict_arima(self):
        """
        ARIMA 모델로 미래 가격을 예측합니다.
        
        Returns:
            numpy.ndarray: 예측된 미래 가격 변화율
        """
        if self.arima_model is None:
            return np.zeros(self.n_forecast)
        
        try:
            # ARIMA 예측 수행
            if isinstance(self.arima_model, pm.arima.ARIMA):
                # pmdarima 패키지 사용 시
                forecast = self.arima_model.predict(n_periods=self.n_forecast)
            else:
                # statsmodels 패키지 사용 시
                forecast = self.arima_model.forecast(steps=self.n_forecast)
                
            self.last_arima_forecast = forecast
            
            # 일별 변화율로 변환
            if len(forecast) > 1:
                forecast_diff = np.diff(np.append(0, forecast))
            else:
                forecast_diff = forecast
                
            return forecast_diff
        except Exception as e:
            print(f"ARIMA 예측 실패: {e}")
            return np.zeros(self.n_forecast)
    
    def predict_volatility(self):
        """
        GARCH 모델로 미래 변동성을 예측합니다.
        
        Returns:
            numpy.ndarray: 예측된 미래 변동성
        """
        if self.garch_model is None or not hasattr(self, 'garch_fit'):
            return np.ones(self.n_forecast) * 0.01  # 기본 변동성 1%
        
        try:
            # GARCH 예측 수행
            forecast = self.garch_fit.forecast(horizon=self.n_forecast)
            # 변동성 (표준편차) 추출
            volatility = np.sqrt(forecast.variance.values[-1, :])
            self.last_volatility_forecast = volatility / 100  # 원래 스케일로 변환
            return self.last_volatility_forecast
        except Exception as e:
            print(f"GARCH 예측 실패: {e}")
            return np.ones(self.n_forecast) * 0.01
    
    def generate_combined_features(self, current_price, other_features=None):
        """
        ARIMA와 GARCH 예측을 결합하여 강화학습 입력 특성을 생성합니다.
        
        Args:
            current_price (float): 현재 가격
            other_features (numpy.ndarray, optional): 기존 특성 벡터
            
        Returns:
            numpy.ndarray: 통합된 예측 및 특성 벡터
        """
        # ARIMA 가격 예측
        price_forecast_diff = self.predict_arima()
        
        # GARCH 변동성 예측
        volatility_forecast = self.predict_volatility()
        
        # 가격 예측을 누적하여 실제 가격 예측으로 변환
        price_forecast = np.ones(self.n_forecast) * current_price
        for i in range(self.n_forecast):
            if i > 0:
                price_forecast[i] = price_forecast[i-1] * (1 + price_forecast_diff[i])
            else:
                price_forecast[i] = current_price * (1 + price_forecast_diff[i])
        
        # 예측된 가격과 변동성을 통합 특성으로 변환
        forecast_features = np.array([
            # 예측된 가격 변화율
            (price_forecast[0] - current_price) / current_price,
            np.mean(price_forecast_diff),
            np.std(price_forecast_diff),
            
            # 예측된 변동성
            volatility_forecast[0],
            np.mean(volatility_forecast),
            np.max(volatility_forecast),
            
            # 추세 방향 (상승/하락)
            1 if np.mean(price_forecast_diff) > 0 else -1,
            
            # 변동성 급증 예상 여부
            1 if np.max(volatility_forecast) > 1.5 * volatility_forecast[0] else 0
        ])
        
        # 다른 특성이 있으면 결합
        if other_features is not None:
            combined_features = np.concatenate([other_features, forecast_features])
            return combined_features
        
        return forecast_features
    
    def calculate_confidence_interval(self, current_price):
        """
        예측 가격의 신뢰 구간을 계산합니다.
        
        Args:
            current_price (float): 현재 가격
            
        Returns:
            tuple: (하한, 상한) 신뢰 구간
        """
        if self.last_arima_forecast is None or self.last_volatility_forecast is None:
            return current_price * 0.98, current_price * 1.02
        
        # 가격 변화 예측
        price_change = self.last_arima_forecast[0]
        # 변동성 예측
        volatility = self.last_volatility_forecast[0]
        
        # 95% 신뢰 구간 계산 (1.96 시그마)
        lower_bound = current_price * (1 + price_change - 1.96 * volatility)
        upper_bound = current_price * (1 + price_change + 1.96 * volatility)
        
        return lower_bound, upper_bound
    
    def get_trend_probability(self):
        """
        ARIMA 모델 기반 추세 방향 확률을 계산합니다.
        
        Returns:
            float: 상승 추세 확률 (0~1 사이 값)
        """
        if self.last_arima_forecast is None:
            return 0.5  # 기본값
            
        # 예측된 가격 변화 방향
        price_changes = self.last_arima_forecast
        
        # 단순하게 상승 예측 비율을 확률로 사용
        up_prob = np.mean(price_changes > 0)
        
        # 크기 반영 상승 확률
        if np.mean(price_changes) > 0:
            return 0.5 + min(0.5, abs(np.mean(price_changes)) * 20)  # 최대 100%
        else:
            return 0.5 - min(0.5, abs(np.mean(price_changes)) * 20)  # 최소 0%
    
    def get_regime_prediction(self):
        """
        미래 시장 레짐을 예측합니다.
        
        Returns:
            str: 예측된 시장 레짐 (추세 상승, 추세 하락, 횡보, 고변동성, 저변동성)
        """
        if self.last_arima_forecast is None or self.last_volatility_forecast is None:
            return "NEUTRAL"
            
        # 추세 방향과 크기
        trend_direction = np.mean(self.last_arima_forecast)
        trend_magnitude = abs(trend_direction)
        
        # 변동성 평균
        avg_volatility = np.mean(self.last_volatility_forecast)
        
        # 레짐 분류
        if avg_volatility > 0.02:  # 일평균 2% 이상 변동 시 고변동성
            return "HIGH_VOLATILITY"
        elif avg_volatility < 0.005:  # 일평균 0.5% 이하 변동 시 저변동성
            return "LOW_VOLATILITY"
        elif trend_magnitude > 0.01:  # 일평균 1% 이상 변화 시 추세
            if trend_direction > 0:
                return "TREND_UP"
            else:
                return "TREND_DOWN"
        else:
            return "RANGE_BOUND" 