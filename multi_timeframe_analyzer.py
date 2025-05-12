import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats

class MultiTimeframeAnalyzer:
    """
    다양한 시간대(1분, 5분, 15분, 1시간, 4시간, 1일)의 데이터를 분석하여
    시간대 간 일관성 있는 신호를 생성하는 클래스
    """
    
    # 타임프레임 상수 정의
    TF_1M = '1m'
    TF_5M = '5m'
    TF_15M = '15m'
    TF_1H = '1h'
    TF_4H = '4h'
    TF_1D = '1d'
    
    # 타임프레임 가중치 (장기 타임프레임에 더 높은 가중치)
    TF_WEIGHTS = {
        TF_1M: 0.05,
        TF_5M: 0.10,
        TF_15M: 0.15,
        TF_1H: 0.20,
        TF_4H: 0.25,
        TF_1D: 0.25
    }
    
    def __init__(self, timeframes=None):
        """
        다중 타임프레임 분석기 초기화
        
        Args:
            timeframes (list): 분석할 타임프레임 목록. 기본값은 모든 타임프레임
        """
        self.timeframes = timeframes or [self.TF_1M, self.TF_5M, self.TF_15M, 
                                         self.TF_1H, self.TF_4H, self.TF_1D]
        self.data = {}  # 각 타임프레임별 데이터를 저장
        self.indicators = {}  # 각 타임프레임별 지표를 저장
        
    def add_timeframe_data(self, timeframe, df):
        """
        특정 타임프레임의 OHLCV 데이터를 추가
        
        Args:
            timeframe (str): 타임프레임 (1m, 5m, 15m, 1h, 4h, 1d)
            df (pandas.DataFrame): OHLCV 데이터 ('open', 'high', 'low', 'close', 'volume' 포함)
        """
        if timeframe not in self.TF_WEIGHTS:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")
        
        self.data[timeframe] = df.copy()
        self._calculate_indicators(timeframe)
    
    def _calculate_indicators(self, timeframe):
        """
        특정 타임프레임에 대한 기술적 지표 계산
        
        Args:
            timeframe (str): 타임프레임
        """
        df = self.data[timeframe]
        
        # 지표를 담을 딕셔너리 초기화
        indicators = {}
        
        # 이동평균 계산
        indicators['sma_20'] = ta.sma(df['close'], length=20)
        indicators['sma_50'] = ta.sma(df['close'], length=50)
        indicators['sma_200'] = ta.sma(df['close'], length=200)
        
        # MACD 계산
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        indicators['macd'] = macd[f'MACD_12_26_9']
        indicators['macd_signal'] = macd[f'MACDs_12_26_9']
        indicators['macd_hist'] = macd[f'MACDh_12_26_9']
        
        # RSI 계산
        indicators['rsi'] = ta.rsi(df['close'], length=14)
        
        # 볼린저 밴드
        bbands = ta.bbands(df['close'], length=20, std=2)
        indicators['bb_upper'] = bbands[f'BBU_20_2.0']
        indicators['bb_middle'] = bbands[f'BBM_20_2.0']
        indicators['bb_lower'] = bbands[f'BBL_20_2.0']
        
        # ATR 계산
        indicators['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # 추세 방향 및 강도
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        indicators['adx'] = adx[f'ADX_14']
        indicators['plus_di'] = adx[f'DMP_14']
        indicators['minus_di'] = adx[f'DMN_14']
        
        # 결과 저장
        self.indicators[timeframe] = indicators
    
    def get_trend_signals(self):
        """
        모든 타임프레임에서 추세 신호를 분석하고 통합 신호 반환
        
        Returns:
            dict: 통합된 추세 신호 (-1: 하락추세, 0: 중립, 1: 상승추세)
        """
        # 각 타임프레임 별 추세 신호
        timeframe_signals = {}
        
        for tf in self.data.keys():
            if tf not in self.indicators:
                continue
                
            ind = self.indicators[tf]
            df = self.data[tf]
            
            # 이동평균 기반 추세 확인
            sma_signal = 0
            if ind['sma_20'].iloc[-1] > ind['sma_50'].iloc[-1]:
                sma_signal = 1
            elif ind['sma_20'].iloc[-1] < ind['sma_50'].iloc[-1]:
                sma_signal = -1
                
            # MACD 기반 추세 확인
            macd_signal = 0
            if ind['macd_hist'].iloc[-1] > 0:
                macd_signal = 1
            elif ind['macd_hist'].iloc[-1] < 0:
                macd_signal = -1
                
            # ADX 기반 추세 강도 및 방향 확인
            adx_signal = 0
            if ind['adx'].iloc[-1] > 25:  # 강한 추세
                if ind['plus_di'].iloc[-1] > ind['minus_di'].iloc[-1]:
                    adx_signal = 1
                else:
                    adx_signal = -1
            
            # 종합 신호
            timeframe_signals[tf] = (sma_signal + macd_signal + adx_signal) / 3
        
        # 가중 평균으로 타임프레임 통합
        weighted_signal = 0
        total_weight = 0
        
        for tf, signal in timeframe_signals.items():
            weight = self.TF_WEIGHTS[tf]
            weighted_signal += signal * weight
            total_weight += weight
            
        if total_weight > 0:
            weighted_signal /= total_weight
            
        # 신호 분류 (-1 ~ 1 범위)
        return {
            'trend_signal': weighted_signal,  # 원시 가중 신호
            'trend_direction': 1 if weighted_signal > 0.3 else (-1 if weighted_signal < -0.3 else 0),  # 이산화된 방향
            'trend_strength': abs(weighted_signal)  # 추세 강도 (0 ~ 1)
        }
    
    def get_support_resistance(self):
        """
        모든 타임프레임에서 지지/저항 레벨을 찾아 통합합니다.
        
        Returns:
            dict: 가장 중요한 지지/저항 레벨
        """
        all_levels = []
        
        # 각 타임프레임별 지지/저항 레벨 식별 (여기서는 볼린저 밴드 사용)
        for tf in self.data.keys():
            if tf not in self.indicators:
                continue
                
            ind = self.indicators[tf]
            df = self.data[tf]
            weight = self.TF_WEIGHTS[tf]
            
            # 볼린저 밴드 기반 저항/지지
            resistance_level = ind['bb_upper'].iloc[-1]
            support_level = ind['bb_lower'].iloc[-1]
            
            all_levels.append((support_level, weight, 'support'))
            all_levels.append((resistance_level, weight, 'resistance'))
            
        # 주요 레벨 선택
        supports = [level for level, _, level_type in all_levels if level_type == 'support']
        resistances = [level for level, _, level_type in all_levels if level_type == 'resistance']
        
        # 평균 레벨 계산
        avg_support = np.mean(supports) if supports else None
        avg_resistance = np.mean(resistances) if resistances else None
        
        return {
            'support': avg_support,
            'resistance': avg_resistance
        }
    
    def get_volatility_metrics(self):
        """
        모든 타임프레임에서 변동성 지표를 분석하고 통합
        
        Returns:
            dict: 통합된 변동성 지표
        """
        volatility_metrics = {}
        
        for tf in self.data.keys():
            if tf not in self.indicators:
                continue
                
            ind = self.indicators[tf]
            df = self.data[tf]
            
            # ATR 기반 변동성
            volatility_metrics[tf] = {
                'atr': ind['atr'].iloc[-1],
                'atr_pct': ind['atr'].iloc[-1] / df['close'].iloc[-1] * 100,  # 가격 대비 ATR %
                'bb_width': (ind['bb_upper'].iloc[-1] - ind['bb_lower'].iloc[-1]) / ind['bb_middle'].iloc[-1] * 100  # BB 폭 %
            }
        
        # 통합 변동성 지표 계산 (가중 평균)
        weighted_atr_pct = 0
        weighted_bb_width = 0
        total_weight = 0
        
        for tf, metrics in volatility_metrics.items():
            weight = self.TF_WEIGHTS[tf]
            weighted_atr_pct += metrics['atr_pct'] * weight
            weighted_bb_width += metrics['bb_width'] * weight
            total_weight += weight
            
        if total_weight > 0:
            weighted_atr_pct /= total_weight
            weighted_bb_width /= total_weight
            
        return {
            'volatility': weighted_atr_pct,  # 주요 변동성 지표
            'bb_width': weighted_bb_width,  # 보조 변동성 지표
            'volatility_regime': 'high' if weighted_atr_pct > 2.0 else ('low' if weighted_atr_pct < 0.5 else 'normal')
        }
    
    def generate_multi_timeframe_signals(self):
        """
        모든 타임프레임에서 종합 트레이딩 신호 생성
        
        Returns:
            dict: 통합된 트레이딩 신호 및 메트릭
        """
        # 충분한 데이터가 있는지 확인
        if not self.data or not self.indicators:
            return {'error': '데이터가 충분하지 않습니다.'}
        
        # 추세, 지지/저항, 변동성 정보 가져오기
        trend_info = self.get_trend_signals()
        sr_levels = self.get_support_resistance()
        volatility_info = self.get_volatility_metrics()
        
        # 현재 가격 (가장 짧은 타임프레임 기준)
        shortest_tf = sorted(self.data.keys(), 
                            key=lambda x: {'1m': 1, '5m': 2, '15m': 3, '1h': 4, '4h': 5, '1d': 6}[x])[0]
        current_price = self.data[shortest_tf]['close'].iloc[-1]
        
        # 매수/매도 신호 생성
        buy_signal = 0  # -1(강한 매도) ~ 1(강한 매수)
        
        # 1. 추세 기반 신호
        buy_signal += trend_info['trend_direction'] * 0.5
        
        # 2. 가격이 지지선 근처면 매수 신호 강화
        if sr_levels['support'] and current_price < sr_levels['support'] * 1.02:
            buy_signal += 0.3
            
        # 3. 가격이 저항선 근처면 매도 신호 강화
        if sr_levels['resistance'] and current_price > sr_levels['resistance'] * 0.98:
            buy_signal -= 0.3
            
        # 4. 변동성 기반 조정
        if volatility_info['volatility_regime'] == 'high':
            buy_signal *= 0.7  # 고변동성에서 신호 약화
            
        # 이산화된 신호
        signal_strength = abs(buy_signal)
        signal_type = None
        
        if buy_signal > 0.5:
            signal_type = 'STRONG_BUY'
        elif buy_signal > 0.2:
            signal_type = 'BUY'
        elif buy_signal < -0.5:
            signal_type = 'STRONG_SELL'
        elif buy_signal < -0.2:
            signal_type = 'SELL'
        else:
            signal_type = 'NEUTRAL'
        
        # 결과 반환
        return {
            'signal': signal_type,
            'signal_strength': signal_strength,
            'raw_signal': buy_signal,
            'trend': trend_info,
            'support_resistance': sr_levels,
            'volatility': volatility_info,
            'price': current_price
        } 