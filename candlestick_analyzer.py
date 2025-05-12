import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행되도록 백엔드 설정
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import logging
from PIL import Image
import torch
from torchvision import transforms, models
from tqdm import tqdm
import glob
import re

class CandlestickAnalyzer:
    def __init__(self, config=None):
        # 설정 저장
        self.config = config or {}
        
        # 설정에서 결과 저장 경로 가져오기
        output_config = self.config.get('output', {})
        self.results_dir = output_config.get('save_dir', 'results/paper1')
        
        # 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger('CandlestickAnalyzer')
        
        # CNN 모델 초기화
        self.cnn_model = None
        self._initialize_cnn_model()
        
    def _initialize_cnn_model(self):
        """CNN 모델 초기화"""
        try:
            # ResNet50 사용
            self.cnn_model = models.resnet50(pretrained=True)
            
            # 분류 레이어 제거 (특징 추출만 사용)
            self.cnn_model = torch.nn.Sequential(*list(self.cnn_model.children())[:-1])
            
            # GPU 사용 여부 확인
            use_gpu = self.config.get('use_gpu', True) and torch.cuda.is_available()
            self.device = torch.device('cuda' if use_gpu else 'cpu')
            self.cnn_model = self.cnn_model.to(self.device)
            
            # 평가 모드로 설정
            self.cnn_model.eval()
            
            self.logger.info(f"CNN 모델 초기화 완료. 장치: {self.device}")
        except Exception as e:
            self.logger.error(f"CNN 모델 초기화 오류: {str(e)}")
    
    def fetch_data(self, symbol='BTCUSDT', timeframe='1h', days_back=30):
        """캔들스틱 데이터 가져오기"""
        try:
            # 샘플 데이터 생성 (실제로는 Binance API 사용)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # 날짜 범위 생성
            date_range = pd.date_range(start=start_date, end=end_date, freq=timeframe)
            
            # 샘플 데이터 생성
            data = []
            last_close = np.random.uniform(40000, 45000)  # 초기 BTC 가격
            
            for date in date_range:
                open_price = last_close
                high_price = open_price * np.random.uniform(1.0, 1.05)
                low_price = open_price * np.random.uniform(0.95, 1.0)
                close_price = np.random.uniform(low_price, high_price)
                volume = np.random.uniform(100, 1000)
                
                data.append({
                    'timestamp': date,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
                last_close = close_price
            
            # 데이터프레임 생성
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"샘플 캔들스틱 데이터 생성 완료: {len(df)} 항목")
            
            # 결과 저장
            if self.results_dir:
                csv_path = os.path.join(self.results_dir, f"{symbol}_{timeframe}_data.csv")
                df.to_csv(csv_path)
                self.logger.info(f"캔들스틱 데이터 저장: {csv_path}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"캔들스틱 데이터 가져오기 오류: {str(e)}")
            return pd.DataFrame()
    
    def generate_candlestick_images(self, data, output_dir=None, window_size=6):
        """캔들스틱 차트 이미지 생성 - 각 이미지에 6개의 캔들스틱 표시"""
        try:
            if output_dir is None and self.results_dir:
                output_dir = os.path.join(self.results_dir, 'images')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 이동 윈도우로 이미지 생성
            images_paths = []
            for i in range(0, len(data) - window_size + 1):
                window_data = data.iloc[i:i+window_size]
                
                # 파일명 생성
                end_date = window_data.index[-1].strftime('%Y%m%d_%H%M%S')
                filename = f"candlestick_{end_date}.png"
                filepath = os.path.join(output_dir, filename)
                
                # mplfiance로 캔들스틱 차트 생성
                mpf.plot(
                    window_data,
                    type='candle',
                    style='yahoo',
                    title=f"BTC/USDT Candlestick Chart (6 candles)",
                    ylabel='Price (USDT)',
                    volume=True,
                    savefig=filepath
                )
                
                images_paths.append(filepath)
            
            self.logger.info(f"{len(images_paths)}개의 캔들스틱 이미지 생성 완료 (각 이미지당 6개 캔들)")
            return images_paths
        
        except Exception as e:
            self.logger.error(f"캔들스틱 이미지 생성 오류: {str(e)}")
            return []
    
    def analyze_patterns(self, chart_path):
        """차트 이미지에서 패턴 분석"""
        try:
            self.logger.info(f"차트 이미지 분석 시작: {chart_path}")
            
            # 차트 경로가 존재하는지 확인
            if not os.path.exists(chart_path):
                self.logger.warning(f"차트 경로가 존재하지 않습니다: {chart_path}")
                return {'prediction': 0.0}  # 중립 신호 반환
            
            # 이미지 파일 목록 가져오기
            image_files = [f for f in os.listdir(chart_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                self.logger.warning(f"분석할 이미지가 없습니다: {chart_path}")
                return {'prediction': 0.0}  # 중립 신호 반환
            
            # 이미지 변환 설정
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # 각 이미지 분석
            predictions = []
            for image_file in tqdm(image_files, desc=f'[{os.path.basename(chart_path)}] 캔들 이미지 분석', position=2, leave=False):
                try:
                    image_path = os.path.join(chart_path, image_file)
                    if not os.path.exists(image_path):
                        self.logger.warning(f"이미지 파일이 존재하지 않습니다: {image_path}")
                        continue
                        
                    with Image.open(image_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_tensor = transform(img).unsqueeze(0).to(self.device)
                    
                    # 모델을 GPU에 올림 (이미 올려져 있으면 영향 없음)
                    self.cnn_model = self.cnn_model.to(self.device)
                    
                    with torch.no_grad():
                        features = self.cnn_model(img_tensor)
                        features = features.squeeze()
                        features = features.cpu().numpy()
                        
                        if features.size > 0:  # 특징이 추출되었는지 확인
                            pred = np.mean(features)
                            predictions.append(pred)
                        else:
                            self.logger.warning(f"이미지에서 특징 추출 실패: {image_path}")
                except Exception as e:
                    self.logger.error(f"이미지 분석 중 오류 발생: {image_path}, {str(e)}")
            
            # 예측 결과가 없는 경우 중립 신호 반환
            if not predictions:
                self.logger.warning("유효한 예측 결과가 없습니다")
                return {'prediction': 0.0}
            
            # 최종 예측 계산
            final_prediction = np.mean(predictions)
            
            # 값의 범위가 너무 작은 경우를 방지하기 위한 처리
            min_pred = np.min(predictions) if predictions else -1
            max_pred = np.max(predictions) if predictions else 1
            
            if max_pred == min_pred:  # 모든 예측값이 동일한 경우
                final_prediction = 0.0  # 중립 신호 반환
            else:
                # -1 ~ 1 범위로 정규화
                final_prediction = 2 * (final_prediction - min_pred) / (max_pred - min_pred + 1e-10) - 1
            
            self.logger.info(f"분석 완료, 예측값: {final_prediction:.4f}")
            return {'prediction': final_prediction}
            
        except Exception as e:
            self.logger.error(f"캔들스틱 패턴 분석 오류: {str(e)}")
            return {'prediction': 0.0}  # 오류 발생 시 중립 신호 반환
    
    def analyze(self, data):
        """캔들스틱 분석 수행"""
        return self.analyze_patterns(data)
        
    def save_charts(self, charts, output_dir=None):
        """차트 저장"""
        if output_dir is None and self.results_dir:
            output_dir = os.path.join(self.results_dir, 'charts')
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, chart in enumerate(charts):
            file_path = os.path.join(output_dir, f'chart_{i}.png')
            chart.savefig(file_path)
            plt.close(chart)
        
        self.logger.info(f"{len(charts)}개 차트가 {output_dir}에 저장되었습니다.")

    def extract_features(self, image_paths, candlestick_data=None):
        """
        캔들스틱 이미지에서 특징 추출
        
        Args:
            image_paths (list): 이미지 파일 경로 리스트
            candlestick_data (DataFrame, optional): 캔들스틱 데이터
            
        Returns:
            list: 추출된 특징 벡터 리스트
        """
        self.logger.info(f"캔들스틱 이미지 {len(image_paths)}개에서 특징 추출 중...")
        
        try:
            # 이미지 변환 설정
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # 특징 추출
            features = []
            with torch.no_grad():
                for image_path in image_paths:
                    try:
                        # 이미지 로드 및 전처리
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0).to(self.device)
                        
                        # 특징 추출
                        feature = self.cnn_model(image_tensor)
                        feature = feature.squeeze().cpu().numpy()
                        
                        features.append(feature)
                    except Exception as e:
                        self.logger.error(f"이미지 {image_path} 처리 중 오류 발생: {str(e)}")
                        # 오류 발생 시 0으로 채워진 특징 벡터 추가
                        features.append(np.zeros(2048))
            
            self.logger.info(f"캔들스틱 이미지에서 {len(features)}개의 특징 추출 완료")
            
            # 결과 저장
            if self.results_dir:
                np.save(os.path.join(self.results_dir, 'candlestick_features.npy'), features)
                self.logger.info(f"캔들스틱 특징 저장 완료: {os.path.join(self.results_dir, 'candlestick_features.npy')}")
            
            return features
        
        except Exception as e:
            self.logger.error(f"캔들스틱 이미지 특징 추출 오류: {str(e)}")
            return []
            
    def get_trading_days(self):
        """
        차트 이미지에서 거래일 목록을 추출하여 반환
        
        Returns:
            list: 거래일 목록 (날짜 문자열)
        """
        try:
            # 설정에서 차트 디렉토리 경로 가져오기
            chart_dir = self.config.get('data', {}).get('chart_dir', 'D:/drl-candlesticks-trader-main1/paper1/data/chart')
            timeframes = self.config.get('data', {}).get('timeframes', ['4h'])
            
            # 첫 번째 시간프레임 선택
            timeframe = timeframes[0]
            
            # 해당 시간프레임의 차트 디렉토리 경로
            timeframe_dir = os.path.join(chart_dir, timeframe)
            
            self.logger.info(f"거래일 목록 추출 중: {timeframe_dir}")
            
            # 디렉토리가 존재하는지 확인
            if not os.path.exists(timeframe_dir):
                self.logger.warning(f"차트 디렉토리가 존재하지 않습니다: {timeframe_dir}")
                return self._get_default_trading_days()
            
            # 이미지 파일 검색
            image_files = []
            for ext in ['png', 'jpg', 'jpeg']:
                pattern = os.path.join(timeframe_dir, f'*.{ext}')
                image_files.extend(glob.glob(pattern))
            
            if not image_files:
                self.logger.warning(f"차트 디렉토리에 이미지 파일이 없습니다: {timeframe_dir}")
                return self._get_default_trading_days()
            
            # 파일 이름에서 날짜 추출
            dates = []
            for file_path in image_files:
                filename = os.path.basename(file_path)
                # 날짜_시간_가격.png 형식에서 날짜 추출
                match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
                if match:
                    date_str = match.group(1)
                    dates.append(date_str)
            
            # 중복 제거 및 정렬
            unique_dates = sorted(list(set(dates)))
            
            self.logger.info(f"추출된 거래일 목록: {len(unique_dates)}일")
            
            # 거래일 캐싱 (추후 get_close_price에서 사용)
            self.trading_days = unique_dates
            self.price_cache = {}
            
            return unique_dates
        
        except Exception as e:
            self.logger.error(f"거래일 목록 추출 오류: {str(e)}")
            return self._get_default_trading_days()
    
    def _get_default_trading_days(self):
        """기본 거래일 목록 생성"""
        # 기본값으로 최근 30일의 가상 데이터 반환
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        return date_strings
    
    def get_close_price(self, date):
        """
        특정 날짜의 종가 반환
        
        Args:
            date (str): 날짜 문자열 (YYYY-MM-DD)
            
        Returns:
            float: 종가
        """
        try:
            # 캐시에 있는지 확인
            if date in self.price_cache:
                return self.price_cache[date]
            
            # 설정에서 차트 디렉토리 경로 가져오기
            chart_dir = self.config.get('data', {}).get('chart_dir', 'D:/drl-candlesticks-trader-main1/paper1/data/chart')
            timeframes = self.config.get('data', {}).get('timeframes', ['4h'])
            
            # 첫 번째 시간프레임 선택
            timeframe = timeframes[0]
            
            # 해당 시간프레임의 차트 디렉토리 경로
            timeframe_dir = os.path.join(chart_dir, timeframe)
            
            # 디렉토리가 존재하는지 확인
            if not os.path.exists(timeframe_dir):
                self.logger.warning(f"차트 디렉토리가 존재하지 않습니다: {timeframe_dir}")
                return self._get_default_price(date)
            
            # 이미지 파일 검색
            pattern = os.path.join(timeframe_dir, f'*{date}*.png')
            image_files = glob.glob(pattern)
            
            if not image_files:
                # 파일을 찾을 수 없는 경우, 이전 날짜의 가격에 약간의 변동을 추가하여 반환
                return self._get_default_price(date)
            
            # 파일 이름에서 가격 추출
            filename = os.path.basename(image_files[0])
            # 날짜_시간_가격.png 형식에서 가격 추출
            match = re.search(r'_(\d+\.\d+)\.png$', filename)
            if match:
                price = float(match.group(1))
            else:
                # 가격을 추출할 수 없는 경우 가상 가격 생성
                price = self._get_default_price(date)
            
            # 가격 캐싱 및 마지막 가격 업데이트
            self.price_cache[date] = price
            self.last_price = price
            
            return price
        
        except Exception as e:
            self.logger.error(f"종가 검색 오류 ({date}): {str(e)}")
            return self._get_default_price(date)
    
    def _get_default_price(self, date):
        """기본 가격 생성"""
        # 이전 가격이 있으면 그 가격에 약간의 변동을 추가
        if hasattr(self, 'last_price') and self.last_price is not None:
            price = self.last_price * np.random.uniform(0.98, 1.02)
        else:
            # 초기 가격 (BTC 가격으로 가정)
            price = 45000.0 * np.random.uniform(0.98, 1.02)
        
        # 가격 캐싱 및 마지막 가격 업데이트
        self.price_cache[date] = price
        self.last_price = price
        
        return price

    def get_pattern_signals(self, date):
        """
        특정 날짜의 캔들스틱 패턴 신호 반환
        
        Args:
            date (str): 날짜 문자열 (YYYY-MM-DD)
            
        Returns:
            float: 패턴 신호 (-1 ~ 1)
        """
        try:
            # 설정에서 차트 디렉토리 경로 가져오기
            chart_dir = self.config.get('data', {}).get('chart_dir', 'D:/drl-candlesticks-trader-main1/paper1/data/chart')
            timeframes = self.config.get('data', {}).get('timeframes', ['4h'])
            
            # 각 시간프레임별 신호 분석
            signals = []
            
            for timeframe in timeframes:
                # 해당 시간프레임의 차트 디렉토리 경로
                timeframe_dir = os.path.join(chart_dir, timeframe)
                
                # 디렉토리가 존재하는지 확인
                if not os.path.exists(timeframe_dir):
                    self.logger.warning(f"차트 디렉토리가 존재하지 않습니다: {timeframe_dir}")
                    signals.append(0.0)
                    continue
                
                # 이미지 파일 검색
                pattern = os.path.join(timeframe_dir, f'*{date}*.png')
                image_files = glob.glob(pattern)
                
                if not image_files:
                    # 파일을 찾을 수 없는 경우, 중립 신호(0) 반환
                    signals.append(0.0)
                    continue
                
                # 각 이미지 분석
                timeframe_signals = []
                for image_path in image_files:
                    try:
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        
                        with Image.open(image_path) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                            img_tensor = transform(img).unsqueeze(0).to(self.device)
                        
                        with torch.no_grad():
                            features = self.cnn_model(img_tensor)
                            features = features.squeeze()
                            features = features.cpu().numpy()
                            
                            if features.size > 0:  # 특징이 추출되었는지 확인
                                pred = np.mean(features)
                                
                                # 신호를 -1 ~ 1 범위로 정규화
                                min_val = np.min(features)
                                max_val = np.max(features)
                                
                                if max_val > min_val:  # 값의 범위가 있는 경우에만 정규화
                                    normalized_pred = 2.0 * (pred - min_val) / (max_val - min_val) - 1.0
                                else:
                                    normalized_pred = 0.0  # 모든 값이 동일하면 중립 신호
                                
                                timeframe_signals.append(normalized_pred)
                            else:
                                self.logger.warning(f"이미지에서 특징 추출 실패: {image_path}")
                    except Exception as e:
                        self.logger.error(f"이미지 분석 중 오류 발생: {image_path}, {str(e)}")
                
                # 해당 시간프레임의 평균 신호 계산
                if timeframe_signals:
                    avg_signal = np.mean(timeframe_signals)
                    signals.append(avg_signal)
                else:
                    signals.append(0.0)
            
            # 모든 시간프레임의 평균 신호 계산
            if signals:
                final_signal = np.mean(signals)
            else:
                final_signal = 0.0
            
            return final_signal
        
        except Exception as e:
            self.logger.error(f"패턴 신호 분석 오류 ({date}): {str(e)}")
            return 0.0  # 오류 발생 시 중립 신호 반환 