import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import re
import json
from collections import Counter
import ast

class SentimentAnalyzer:
    def __init__(self, config=None):
        # 설정 저장
        self.config = config or {}
        
        # 설정에서 결과 저장 경로 가져오기
        output_config = self.config.get('output', {})
        self.results_dir = output_config.get('save_dir', 'results/paper1')
        
        # 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger('SentimentAnalyzer')
        
        # 감성 분석 모델 설정
        sentiment_config = self.config.get('sentiment_model', {})
        
        # 강세/약세 키워드 (예시)
        self.bullish_keywords = ['buy', 'bullish', 'upward', 'growth', 'positive', 'rise', 'rising',
                                'increase', 'increasing', 'gains', 'profit', 'rally', 'boom', 'success']
        self.bearish_keywords = ['sell', 'bearish', 'downward', 'decline', 'negative', 'fall', 'falling',
                                 'decrease', 'decreasing', 'loses', 'loss', 'crash', 'bust', 'recession']
    
    def fetch_news(self, symbol='BTC', days_back=7):
        """뉴스 데이터 가져오기"""
        try:
            # 샘플 뉴스 데이터 생성 (실제로는 API 사용)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # 샘플 제목과 내용
            sample_titles = [
                "Bitcoin Price Surges to New Highs as Institutional Interest Grows",
                "Market Analysis: Crypto Assets Show Bullish Patterns",
                "Bitcoin Faces Resistance at $50,000, Analysts Predict Correction",
                "Regulatory Concerns Cause Crypto Market Volatility",
                "Investors Remain Cautious as Bitcoin Struggles to Break Resistance",
                "Altcoins Outperform Bitcoin in Recent Trading Sessions",
                "Technical Analysis Suggests Bearish Trend for Major Cryptocurrencies",
                "Bitcoin Mining Difficulty Hits New Record, Network Security Strengthens",
                "Institutional Adoption of Cryptocurrencies Continues to Expand",
                "Market Sentiment Improves Following Positive Regulatory News"
            ]
            
            sample_contents = [
                "Bitcoin has reached a new all-time high as institutional investors continue to show interest in the cryptocurrency.",
                "Technical analysts are pointing to bullish patterns forming in the charts of major crypto assets.",
                "Despite recent gains, Bitcoin is facing strong resistance at the $50,000 level, with some analysts predicting a potential correction.",
                "Recent regulatory announcements have caused significant volatility in the cryptocurrency market.",
                "Traders and investors remain cautious as Bitcoin struggles to break through key resistance levels.",
                "Several alternative cryptocurrencies have outperformed Bitcoin in recent trading sessions.",
                "According to technical analysis indicators, major cryptocurrencies might be entering a bearish trend.",
                "The difficulty of Bitcoin mining has reached a new record, indicating stronger network security.",
                "More institutional investors are adding cryptocurrencies to their portfolios.",
                "Market sentiment has improved following positive news regarding cryptocurrency regulations."
            ]
            
            # 날짜 범위 생성
            date_range = pd.date_range(start=start_date, end=end_date, periods=len(sample_titles))
            
            # 데이터 생성
            news_data = []
            for i, date in enumerate(date_range):
                news_data.append({
                    'date': date,
                    'title': sample_titles[i % len(sample_titles)],
                    'content': sample_contents[i % len(sample_contents)],
                    'source': f"News Source {i+1}"
                })
            
            # 데이터프레임 생성
            df = pd.DataFrame(news_data)
            
            self.logger.info(f"샘플 뉴스 데이터 생성 완료: {len(df)} 항목")
            
            # 결과 저장
            if self.results_dir:
                csv_path = os.path.join(self.results_dir, f"{symbol}_news_data.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"뉴스 데이터 저장: {csv_path}")
            
            return df
        
        except Exception as e:
            self.logger.error(f"뉴스 데이터 가져오기 오류: {str(e)}")
            return pd.DataFrame()
    
    def analyze_sentiment(self, news_data):
        """뉴스 감성분석: sentiment 컬럼의 polarity/class 값 집계"""
        try:
            if news_data.empty or 'sentiment' not in news_data.columns:
                return {
                    'overall_sentiment': 'neutral',
                    'sentiment_score': 0.0,
                    'bullish_ratio': 0.0,
                    'bearish_ratio': 0.0,
                    'neutral_ratio': 1.0
                }
            polarities = []
            class_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
            for s in news_data['sentiment']:
                try:
                    sent = ast.literal_eval(s)
                    polarities.append(sent.get('polarity', 0.0))
                    c = sent.get('class', 'neutral')
                    if c in class_counts:
                        class_counts[c] += 1
                    else:
                        class_counts['neutral'] += 1
                except:
                    polarities.append(0.0)
                    class_counts['neutral'] += 1
            avg_polarity = sum(polarities) / len(polarities) if polarities else 0.0
            total = sum(class_counts.values())
            bullish_ratio = class_counts['positive'] / total if total else 0.0
            bearish_ratio = class_counts['negative'] / total if total else 0.0
            neutral_ratio = class_counts['neutral'] / total if total else 1.0
            # 종합 감성 판단
            if avg_polarity > 0.1:
                overall_sentiment = 'bullish'
            elif avg_polarity < -0.1:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'neutral'
            return {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': avg_polarity,
                'bullish_ratio': bullish_ratio,
                'bearish_ratio': bearish_ratio,
                'neutral_ratio': neutral_ratio
            }
        except Exception as e:
            self.logger.error(f"뉴스 감성 분석 오류: {str(e)}")
            return {
                'overall_sentiment': 'neutral',
                'sentiment_score': 0.0,
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0,
                'neutral_ratio': 1.0
            }
    
    def extract_features(self, news_data):
        """
        뉴스 데이터에서 특징 추출
        - 실제로는 LSTM, BERT 등의 모델 사용
        - 여기서는 간단한 예시로 랜덤 특징 생성
        """
        self.logger.info(f"뉴스 데이터 {len(news_data)}개에서 특징 추출 중...")
        
        # 여기서는 간단히 랜덤 특징 생성
        # 실제로는 딥러닝 모델로 텍스트 특징 추출
        features = []
        for _ in range(len(news_data)):
            # 128차원의 랜덤 특징 벡터 생성
            feature = np.random.randn(128)
            features.append(feature)
        
        self.logger.info(f"뉴스 감성 특징 추출 완료: {len(features)}개")
        return features
        
    def analyze(self, text):
        """텍스트 감성 분석 수행"""
        # 텍스트를 데이터프레임으로 변환
        if isinstance(text, str):
            news_data = pd.DataFrame([{'title': text, 'content': text}])
        else:
            news_data = text
            
        return self.analyze_sentiment(news_data)
        
    def save_data(self, data, filename=None):
        """데이터 저장"""
        if not self.results_dir:
            self.logger.warning("데이터를 저장할 디렉토리가 지정되지 않았습니다.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_data_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            # 데이터 형식에 따라 저장 방식 변경
            if isinstance(data, pd.DataFrame):
                if filename.endswith('.csv'):
                    data.to_csv(filepath, index=False)
                else:
                    data.to_json(filepath, orient='records', indent=4)
            elif isinstance(data, dict) or isinstance(data, list):
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=4)
            else:
                with open(filepath, 'w') as f:
                    f.write(str(data))
            
            self.logger.info(f"데이터가 저장되었습니다: {filepath}")
            return filepath
        
        except Exception as e:
            self.logger.error(f"데이터 저장 중 오류 발생: {str(e)}")
            return None
            
    def get_sentiment_signals(self, date):
        """
        특정 날짜의 뉴스 감성 신호 반환
        
        Args:
            date (str): 날짜 문자열 (YYYY-MM-DD)
            
        Returns:
            float: 감성 신호 (-1 ~ 1)
        """
        try:
            # 설정에서 뉴스 파일 경로 가져오기
            news_file = self.config.get('data', {}).get('news_file', '')
            
            if not news_file or not os.path.exists(news_file):
                self.logger.warning(f"뉴스 파일을 찾을 수 없습니다: {news_file}")
                # 뉴스 파일이 없는 경우 랜덤 신호 반환
                return np.random.uniform(-0.3, 0.3)
            
            # 날짜 기준으로 필터링 (앞뒤로 3일 윈도우)
            try:
                # 날짜를 datetime 객체로 변환
                target_date = datetime.strptime(date, '%Y-%m-%d')
                
                # 윈도우 설정 (앞뒤로 3일)
                start_date = target_date - timedelta(days=3)
                end_date = target_date + timedelta(days=3)
                
                # 날짜 형식 문자열로 변환
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')
                
                # 캐싱을 위한 날짜 범위 키
                date_range_key = f"{start_date_str}_{end_date_str}"
                
                # 캐시에 저장된 결과가 있는지 확인
                if hasattr(self, 'sentiment_cache') and date_range_key in self.sentiment_cache:
                    return self.sentiment_cache[date_range_key]
                
                # 뉴스 데이터 로드
                try:
                    news_df = pd.read_csv(news_file)
                except Exception as e:
                    self.logger.error(f"뉴스 파일 로드 오류: {str(e)}")
                    # 뉴스 파일 로드 오류 시 랜덤 신호 반환
                    return np.random.uniform(-0.3, 0.3)
                
                # 날짜 컬럼 확인 및 형식 변환
                date_column = None
                for col in news_df.columns:
                    if 'date' in col.lower():
                        date_column = col
                        break
                
                if date_column is None:
                    self.logger.warning("뉴스 데이터에 날짜 컬럼이 없습니다.")
                    return np.random.uniform(-0.3, 0.3)
                
                # 날짜 형식 변환
                try:
                    news_df[date_column] = pd.to_datetime(news_df[date_column])
                except:
                    self.logger.warning("뉴스 데이터의 날짜 형식 변환에 실패했습니다.")
                    return np.random.uniform(-0.3, 0.3)
                
                # 날짜 기준으로 필터링
                filtered_news = news_df[(news_df[date_column] >= start_date) & 
                                       (news_df[date_column] <= end_date)]
                
                if filtered_news.empty:
                    self.logger.warning(f"날짜 범위({start_date_str} ~ {end_date_str})에 일치하는 뉴스가 없습니다.")
                    return 0.0  # 뉴스가 없는 경우 중립 신호
                
                # 텍스트 컬럼 확인
                text_columns = []
                for col in filtered_news.columns:
                    if col.lower() in ['title', 'headline', 'content', 'text', 'body']:
                        text_columns.append(col)
                
                if not text_columns:
                    self.logger.warning("뉴스 데이터에 텍스트 컬럼이 없습니다.")
                    return np.random.uniform(-0.3, 0.3)
                
                # 감성 분석을 위한 텍스트 추출
                sentiment_scores = []
                for _, row in filtered_news.iterrows():
                    # 각 텍스트 컬럼에 대해 감성 분석 수행
                    text = ' '.join([str(row[col]) for col in text_columns if pd.notna(row[col])])
                    
                    # 텍스트 기반 감성 분석
                    bullish_score = 0
                    bearish_score = 0
                    
                    # 강세/약세 키워드 검색
                    for keyword in self.bullish_keywords:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                            bullish_score += 1
                    
                    for keyword in self.bearish_keywords:
                        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                            bearish_score += 1
                    
                    # 최종 감성 점수 계산 (-1 ~ 1)
                    if bullish_score > bearish_score:
                        score = min(1.0, (bullish_score - bearish_score) / max(1, len(self.bullish_keywords) / 5))
                    elif bearish_score > bullish_score:
                        score = max(-1.0, (bullish_score - bearish_score) / max(1, len(self.bearish_keywords) / 5))
                    else:
                        score = 0.0
                    
                    sentiment_scores.append(score)
                
                # 평균 감성 점수 계산
                if sentiment_scores:
                    avg_sentiment = np.mean(sentiment_scores)
                else:
                    avg_sentiment = 0.0
                
                # 감성 점수 캐싱
                if not hasattr(self, 'sentiment_cache'):
                    self.sentiment_cache = {}
                
                self.sentiment_cache[date_range_key] = avg_sentiment
                
                return avg_sentiment
                
            except ValueError:
                self.logger.error(f"날짜 형식 오류: {date}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"감성 신호 분석 오류 ({date}): {str(e)}")
            return 0.0  # 오류 발생 시 중립 신호 반환 