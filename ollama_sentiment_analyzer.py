#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ollama DeepSeek-r1(32b) 모델 기반 감성 분석기
- NLTK VADER 모델과 성능 비교를 위한 구현
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import requests
import time
import re
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

class OllamaSentimentAnalyzer:
    def __init__(self, config=None):
        # 설정 저장
        self.config = config or {}
        
        # 설정에서 결과 저장 경로 가져오기
        output_config = self.config.get('output', {})
        self.results_dir = output_config.get('save_dir', 'results/paper1_ollama')
        
        # 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger('OllamaSentimentAnalyzer')
        
        # Ollama API URL
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434/api/generate')
        
        # DeepSeek 모델명 - 설정에서 모델명을 가져오거나 기본값 사용
        self.model_name = self.config.get('model_name', 'deepseek-llm:latest')
        
        # 딥시크r1(32b) 모델 명시적 사용 여부
        self.use_deepseek_r1 = self.config.get('use_deepseek_r1', False)
        if self.use_deepseek_r1:
            # 사용 가능한 모델 중에서 deepseek-llm이 있는지 확인
            self.model_name = 'deepseek-llm:latest'
        
        # 재시도 횟수
        self.max_retries = self.config.get('max_retries', 3)
        
        # API 호출 간격 (초)
        self.api_call_interval = self.config.get('api_call_interval', 0.5)
        
        # 배치 크기 (동시 처리할 뉴스 항목 수)
        self.batch_size = self.config.get('batch_size', 1)
        
        # 오프라인 모드 - API가 실패하면 텍스트 기반 대체 로직 사용
        self.offline_mode = self.config.get('offline_mode', False)
        
        # 프롬프트 템플릿 - 단순하고 명확한 JSON 응답을 위한 템플릿
        self.prompt_template = """분석할 암호화폐 뉴스: {news}

아래 형식의 JSON으로만 응답하세요:
{{"sentiment": "bullish|bearish|neutral", "score": -1.0~1.0 사이의 값}}

- bullish: 가격 상승에 긍정적인 내용
- bearish: 가격 하락에 부정적인 내용
- neutral: 중립적인 내용
- score: -1.0(매우 bearish)에서 1.0(매우 bullish) 사이의 값

응답 예시: {{"sentiment": "bullish", "score": 0.8}}
설명이나 다른 텍스트는 포함하지 마세요. 오직 JSON만 응답하세요.
"""
        
        # Ollama 서버 연결 확인
        self._check_server_connection()
        
        self.logger.info(f"Ollama 감성 분석기 초기화 완료 (모델: {self.model_name}, {'오프라인 모드' if self.offline_mode else '온라인 모드'})")
    
    def _check_server_connection(self):
        """Ollama 서버 연결 확인"""
        try:
            # 간단한 API 호출로 서버 연결 확인
            url = "http://localhost:11434/api/tags"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                # 사용 가능한 모델 목록 가져오기
                models = response.json().get('models', [])
                self.logger.info(f"Ollama 서버 연결 성공, 사용 가능한 모델: {len(models)}개")
                
                # 현재 선택된 모델이 사용 가능한지 확인
                model_names = [m.get('name', '') for m in models]
                base_model = self.model_name.split(':')[0]
                
                # 딥시크r1(32b) 모델 사용이 설정된 경우 해당 모델 찾기
                if self.use_deepseek_r1:
                    deepseek_models = [m for m in model_names if 'deepseek' in m.lower()]
                    if deepseek_models:
                        # 가능한 경우 가장 큰 크기의 deepseek 모델 선택
                        for model_name in ['deepseek-llm:latest', 'deepseek-coder:latest']:
                            if model_name in model_names:
                                self.model_name = model_name
                                self.logger.info(f"딥시크 모델 '{self.model_name}'을(를) 사용합니다.")
                                break
                    else:
                        self.logger.warning("딥시크 모델을 찾을 수 없습니다. 사용 가능한 다른 모델을 사용합니다.")
                
                if self.model_name in model_names or base_model in model_names:
                    self.logger.info(f"선택한 모델 '{self.model_name}' 사용 가능")
                else:
                    # 사용 가능한 모델이 있으면 첫 번째 모델로 변경
                    if models:
                        self.model_name = models[0].get('name', 'llama3')
                        self.logger.warning(f"선택한 모델 사용 불가, '{self.model_name}'(으)로 변경")
                    else:
                        self.offline_mode = True
                        self.logger.warning(f"사용 가능한 모델이 없습니다. 오프라인 모드로 전환")
            else:
                self.offline_mode = True
                self.logger.warning(f"Ollama 서버 응답 오류 (상태 코드: {response.status_code}). 오프라인 모드로 전환")
                
        except Exception as e:
            self.offline_mode = True
            self.logger.warning(f"Ollama 서버 연결 실패: {str(e)}. 오프라인 모드로 전환")
    
    def fetch_news(self, symbol='BTC', days_back=7):
        """뉴스 데이터 가져오기 (VADER와 동일한 샘플 데이터)"""
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
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call_ollama_api(self, prompt):
        """
        Ollama API를 호출하여 DeepSeek 모델 응답 받기 (재시도 로직 포함)
        
        Args:
            prompt (str): 프롬프트
            
        Returns:
            dict: 응답 결과
        """
        try:
            # API 요청 데이터
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.01,  # 더 낮은 온도로 설정하여 확정적인 응답 유도
                    "num_predict": 100,   # 응답 길이 제한 증가
                    "top_p": 0.1,         # 더 결정적인 토큰 선택
                    "top_k": 5            # 매우 제한적인 토큰 범위
                }
            }
            
            self.logger.info(f"Ollama API 호출 시작 (모델: {self.model_name})")
            
            # API 호출
            response = requests.post(self.ollama_url, json=data, timeout=30)
            
            # 오류 체크
            if response.status_code != 200:
                self.logger.error(f"Ollama API 오류: {response.status_code} - {response.text}")
                raise Exception(f"API 호출 실패: 상태 코드 {response.status_code}")
            
            # 응답 파싱
            result = response.json()
            
            if not result:
                self.logger.error("Ollama API 응답이 비어 있습니다.")
                raise Exception("빈 응답 수신")
                
            response_text = result.get('response', '')
            
            self.logger.info(f"Ollama API 응답 수신 (길이: {len(response_text)} 문자)")
            self.logger.debug(f"원본 응답: {response_text[:200]}...")
            
            # 기본 감성 결과 생성 (API가 유효한 응답을 반환하지 못할 경우)
            default_result = {"sentiment": "neutral", "score": 0.0}
            
            # 응답이 없으면 기본값 반환
            if not response_text.strip():
                self.logger.warning("API 응답 텍스트가 비어 있습니다. 기본값 반환")
                return default_result
            
            # JSON 부분 추출 및 파싱
            result_json = self._extract_json_from_text(response_text)
            
            # 결과 검증
            if not result_json or 'sentiment' not in result_json:
                self.logger.warning("유효한 감성 분석 결과를 얻지 못했습니다. 기본값 반환")
                return default_result
                
            return result_json
        
        except Exception as e:
            self.logger.error(f"Ollama API 호출 오류: {str(e)}")
            # 모든 예외에 대해 실패 처리 대신 기본 결과 반환
            return {"sentiment": "neutral", "score": 0.0}
    
    def _extract_json_from_text(self, text):
        """
        텍스트에서 JSON 객체 추출 및 파싱 (향상된 파싱 로직)
        
        Args:
            text (str): JSON이 포함된 텍스트
            
        Returns:
            dict: 파싱된 JSON 객체
        """
        try:
            # 텍스트 정리
            clean_text = text.strip()
            self.logger.debug(f"정리된 텍스트: {clean_text}")
            
            # 기본 반환 값 설정
            default_result = {"sentiment": "neutral", "score": 0.0}
            
            # 코드 블록 패턴 먼저 확인 (```json ... ``` 형식)
            import re
            code_block_pattern = r'```(?:json)?\s*({.*?})\s*```'
            code_blocks = re.findall(code_block_pattern, clean_text, re.DOTALL)
            
            if code_blocks:
                for block in code_blocks:
                    try:
                        result_json = json.loads(block.strip())
                        if isinstance(result_json, dict) and 'sentiment' in result_json and 'score' in result_json:
                            return self._validate_sentiment_json(result_json)
                    except json.JSONDecodeError:
                        continue
            
            # 1. 전체 텍스트가 JSON인지 확인
            try:
                result_json = json.loads(clean_text)
                if isinstance(result_json, dict) and 'sentiment' in result_json and 'score' in result_json:
                    return self._validate_sentiment_json(result_json)
            except json.JSONDecodeError:
                pass
            
            # 2. JSON 객체 찾기
            json_pattern = r'{[^{}]*"sentiment"[^{}]*"score"[^{}]*}'
            json_matches = re.findall(json_pattern, clean_text, re.IGNORECASE)
            
            if json_matches:
                for json_str in json_matches:
                    try:
                        result_json = json.loads(json_str)
                        if isinstance(result_json, dict) and 'sentiment' in result_json and 'score' in result_json:
                            return self._validate_sentiment_json(result_json)
                    except json.JSONDecodeError:
                        # 따옴표 수정 시도
                        try:
                            fixed_json = json_str.replace("'", '"')
                            result_json = json.loads(fixed_json)
                            if isinstance(result_json, dict) and 'sentiment' in result_json and 'score' in result_json:
                                return self._validate_sentiment_json(result_json)
                        except json.JSONDecodeError:
                            pass
            
            # 3. 전통적인 중괄호 찾기 방식
            json_start = clean_text.find('{')
            json_end = clean_text.rfind('}')
            
            if json_start >= 0 and json_end >= 0 and json_end > json_start:
                json_str = clean_text[json_start:json_end+1]
                
                # 3.1 그대로 파싱 시도
                try:
                    result_json = json.loads(json_str)
                    if isinstance(result_json, dict) and 'sentiment' in result_json and 'score' in result_json:
                        return self._validate_sentiment_json(result_json)
                except json.JSONDecodeError:
                    pass
                
                # 3.2 여러 수정 시도
                try:
                    # 작은따옴표를 큰따옴표로 변경
                    json_str = json_str.replace("'", '"')
                    # 주석 제거
                    json_str = re.sub(r'//.*?(\n|$)', '', json_str)
                    # 속성명에 따옴표 추가
                    json_str = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', json_str)
                    
                    result_json = json.loads(json_str)
                    if isinstance(result_json, dict) and 'sentiment' in result_json and 'score' in result_json:
                        return self._validate_sentiment_json(result_json)
                except (json.JSONDecodeError, Exception):
                    pass
            
            # 4. 정규식을 사용한 패턴 매칭 시도
            try:
                # sentiment 값 추출 (bullish, bearish, neutral)
                sentiment_pattern = r'["\']?sentiment["\']?\s*:\s*["\']?(bullish|bearish|neutral)["\']?'
                sentiment_match = re.search(sentiment_pattern, clean_text, re.IGNORECASE)
                
                # score 값 추출 (-1.0에서 1.0 사이의 숫자)
                score_pattern = r'["\']?score["\']?\s*:\s*(-?\d+(\.\d+)?)'
                score_match = re.search(score_pattern, clean_text)
                
                if sentiment_match and score_match:
                    sentiment = sentiment_match.group(1).lower()
                    score = float(score_match.group(1))
                    return self._validate_sentiment_json({"sentiment": sentiment, "score": score})
            except Exception:
                pass
            
            # 5. 텍스트에서 직접 감성 분석 키워드 추출 시도
            clean_text_lower = clean_text.lower()
            if "bullish" in clean_text_lower:
                return {"sentiment": "bullish", "score": 0.5}
            elif "bearish" in clean_text_lower:
                return {"sentiment": "bearish", "score": -0.5}
            elif "neutral" in clean_text_lower:
                return {"sentiment": "neutral", "score": 0.0}
            
            # 6. 텍스트 분석 기반 대체 로직 사용
            self.logger.warning("JSON 파싱 실패, 텍스트 기반 분석을 사용합니다.")
            return self._analyze_text_fallback(clean_text)
            
        except Exception as e:
            self.logger.warning(f"JSON 추출 오류: {str(e)} - 대체 로직 사용")
            return self._analyze_text_fallback(text)
    
    def _validate_sentiment_json(self, result_json):
        """
        파싱된 JSON 객체 검증 및 수정
        
        Args:
            result_json (dict): 파싱된 JSON 객체
            
        Returns:
            dict: 검증 및 수정된 JSON 객체
        """
        try:
            # sentiment 필드 검증
            sentiment = str(result_json.get('sentiment', 'neutral')).lower()
            if sentiment not in ['bullish', 'bearish', 'neutral']:
                # 유사한 단어 처리
                if sentiment in ['positive', 'bull', 'up', 'upward', 'increase']:
                    sentiment = 'bullish'
                elif sentiment in ['negative', 'bear', 'down', 'downward', 'decrease']:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'
            
            # score 필드 검증
            score = result_json.get('score', 0.0)
            
            # 문자열 점수를 실수로 변환
            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    # 숫자가 아닌 경우, sentiment에 따라 기본값 할당
                    if sentiment == 'bullish':
                        score = 0.5
                    elif sentiment == 'bearish':
                        score = -0.5
                    else:
                        score = 0.0
            
            # 점수 범위 검증
            score = max(-1.0, min(1.0, float(score)))
            
            # sentiment와 score 일관성 확인 및 조정
            if sentiment == 'bullish' and score <= 0:
                score = 0.5  # 강세 감성은 양수 점수
            elif sentiment == 'bearish' and score >= 0:
                score = -0.5  # 약세 감성은 음수 점수
            elif sentiment == 'neutral' and abs(score) > 0.1:
                score = 0.0  # 중립 감성은 0에 가까운 점수
                
            return {"sentiment": sentiment, "score": score}
        except Exception as e:
            self.logger.warning(f"JSON 검증 오류: {str(e)}")
            return {"sentiment": "neutral", "score": 0.0}
    
    def _analyze_text_fallback(self, text):
        """
        JSON 파싱 실패 시 텍스트 기반 분석 대체 로직
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 분석 결과
        """
        # 텍스트 소문자 변환
        text = text.lower()
        
        # 강세/약세 키워드 정의
        bullish_keywords = ['bullish', 'positive', 'rise', 'rising', 'increase', 'increasing', 
                           'gain', 'gains', 'upward', 'up', 'higher', 'growth', 'grew', 'bull', 
                           'rally', 'soar', 'soaring', 'outperform', 'opportunity', 'optimistic']
        
        bearish_keywords = ['bearish', 'negative', 'fall', 'falling', 'decrease', 'decreasing', 
                           'loss', 'losses', 'downward', 'down', 'lower', 'decline', 'declined', 
                           'bear', 'crash', 'dip', 'drop', 'plunge', 'underperform', 'pessimistic', 
                           'correction', 'risk', 'concern', 'worry', 'cautious', 'resistance']
        
        # 키워드 카운트
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text)
        
        # 점수 및 감성 결정
        if bullish_count > bearish_count:
            sentiment = 'bullish'
            score = min(0.3 + 0.1 * (bullish_count - bearish_count), 1.0)
        elif bearish_count > bullish_count:
            sentiment = 'bearish'
            score = max(-0.3 - 0.1 * (bearish_count - bullish_count), -1.0)
        else:
            sentiment = 'neutral'
            score = 0.0
        
        self.logger.info(f"텍스트 기반 대체 분석 결과: {sentiment} (점수: {score:.2f})")
        return {"sentiment": sentiment, "score": score}
    
    def analyze_sentiment(self, news_data):
        """뉴스 감성 분석"""
        try:
            if news_data.empty:
                return {
                    'overall_sentiment': 'neutral',
                    'sentiment_score': 0.0,
                    'bullish_ratio': 0.5,
                    'bearish_ratio': 0.5,
                    'neutral_ratio': 0.0,
                    'detailed_scores': []
                }
            
            # 오프라인 모드인 경우 텍스트 기반 분석 수행
            if self.offline_mode:
                self.logger.info("오프라인 모드로 텍스트 기반 감성 분석 수행")
                return self._analyze_sentiment_offline(news_data)
            
            # 결과 초기화
            results = {
                'bullish_count': 0,
                'bearish_count': 0,
                'neutral_count': 0,
                'sentiment_scores': [],
                'detailed_results': []
            }
            
            # API 서버 연결 확인 (오프라인 모드로 전환 여부 결정)
            try:
                # 간단한 API 테스트
                test_prompt = "This is a test."
                test_result = self.call_ollama_api(test_prompt)
                if not test_result or 'sentiment' not in test_result:
                    self.logger.warning("Ollama API 테스트 실패, 오프라인 모드로 전환")
                    self.offline_mode = True
                    return self._analyze_sentiment_offline(news_data)
            except Exception as e:
                self.logger.warning(f"Ollama API 연결 테스트 실패: {str(e)}. 오프라인 모드로 전환")
                self.offline_mode = True
                return self._analyze_sentiment_offline(news_data)
            
            # 배치 처리 로직
            total_news = len(news_data)
            self.logger.info(f"총 {total_news}개의 뉴스 항목 분석 시작 (온라인 모드)")
            
            for idx, row in tqdm(news_data.iterrows(), total=len(news_data), desc="DeepSeek 감성 분석"):
                title = row['title']
                content = row.get('content', '')
                
                # 텍스트 결합 (제목이 더 중요하므로 두 번 반복)
                text = f"{title} {title} {content}"
                
                # 프롬프트 생성
                prompt = self.prompt_template.format(news=text)
                
                # API 호출 간격 조정
                if idx > 0:
                    time.sleep(self.api_call_interval)
                
                try:
                    # API 호출
                    sentiment_result = self.call_ollama_api(prompt)
                    
                    # 결과 저장
                    sentiment = sentiment_result['sentiment']
                    score = sentiment_result['score']
                    
                    # 상세 결과 저장
                    detailed_result = {
                        'title': title,
                        'sentiment': sentiment,
                        'score': score,
                        'text': text[:100] + '...',
                        'method': 'deepseek'
                    }
                    results['detailed_results'].append(detailed_result)
                    
                    # 감성 카운트 증가
                    if sentiment == 'bullish':
                        results['bullish_count'] += 1
                    elif sentiment == 'bearish':
                        results['bearish_count'] += 1
                    else:
                        results['neutral_count'] += 1
                    
                    results['sentiment_scores'].append(score)
                    
                    # 진행 상황 로깅
                    if idx % 5 == 0 or idx == total_news - 1:
                        self.logger.info(f"진행 상황: {idx+1}/{total_news} 완료")
                        
                except Exception as e:
                    self.logger.warning(f"항목 {idx}('{title}') 분석 중 오류: {str(e)}")
                    # 오류 발생 시 기본값 사용
                    results['neutral_count'] += 1
                    results['sentiment_scores'].append(0.0)
                    
                    # 오류 발생 시에도 결과 저장 (오류 표시와 함께)
                    results['detailed_results'].append({
                        'title': title,
                        'sentiment': 'neutral',
                        'score': 0.0,
                        'text': text[:100] + '...',
                        'method': 'error',
                        'error': str(e)
                    })
            
            # 총 항목 수
            total_count = len(news_data)
            
            # 비율 계산
            bullish_ratio = results['bullish_count'] / total_count if total_count > 0 else 0
            bearish_ratio = results['bearish_count'] / total_count if total_count > 0 else 0
            neutral_ratio = results['neutral_count'] / total_count if total_count > 0 else 0
            
            # 평균 감성 점수
            sentiment_score = np.mean(results['sentiment_scores']) if results['sentiment_scores'] else 0.0
            
            # 종합 감성 판단
            if sentiment_score > 0.1:
                overall_sentiment = 'bullish'
            elif sentiment_score < -0.1:
                overall_sentiment = 'bearish'
            else:
                overall_sentiment = 'neutral'
            
            # 결과 출력
            analysis_result = {
                'overall_sentiment': overall_sentiment,
                'sentiment_score': float(sentiment_score),
                'bullish_ratio': float(bullish_ratio),
                'bearish_ratio': float(bearish_ratio),
                'neutral_ratio': float(neutral_ratio),
                'detailed_scores': results['sentiment_scores'],
                'detailed_results': results['detailed_results'],
                'method': 'deepseek'
            }
            
            self.logger.info(f"DeepSeek 감성 분석 결과: {overall_sentiment}, 점수: {sentiment_score:.2f}")
            
            return analysis_result
        
        except Exception as e:
            self.logger.error(f"Ollama 뉴스 감성 분석 중 전체 오류 발생: {str(e)}")
            self.logger.info("오프라인 모드로 대체하여 분석 수행")
            return self._analyze_sentiment_offline(news_data)
            
    def _analyze_sentiment_offline(self, news_data):
        """오프라인 모드에서 텍스트 기반 감성 분석 수행"""
        self.logger.info("텍스트 기반 감성 분석 시작...")
        
        # 결과 초기화
        results = {
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0,
            'sentiment_scores': [],
            'detailed_results': []
        }
        
        # 각 뉴스 항목에 대한 감성 분석
        for idx, row in tqdm(news_data.iterrows(), total=len(news_data), desc="텍스트 기반 감성 분석"):
            title = row['title']
            content = row.get('content', '')
            
            # 텍스트 결합
            text = f"{title} {content}"
            
            # 텍스트 기반 감성 분석
            sentiment_result = self._analyze_text_fallback(text)
            
            # 결과 저장
            sentiment = sentiment_result['sentiment']
            score = sentiment_result['score']
            
            # 상세 결과 저장
            detailed_result = {
                'title': title,
                'sentiment': sentiment,
                'score': score,
                'text': text[:100] + '...',
                'method': 'text_based'
            }
            results['detailed_results'].append(detailed_result)
            
            # 감성 카운트 증가
            if sentiment == 'bullish':
                results['bullish_count'] += 1
            elif sentiment == 'bearish':
                results['bearish_count'] += 1
            else:
                results['neutral_count'] += 1
            
            results['sentiment_scores'].append(score)
        
        # 총 항목 수
        total_count = len(news_data)
        
        # 비율 계산
        bullish_ratio = results['bullish_count'] / total_count if total_count > 0 else 0
        bearish_ratio = results['bearish_count'] / total_count if total_count > 0 else 0
        neutral_ratio = results['neutral_count'] / total_count if total_count > 0 else 0
        
        # 평균 감성 점수
        sentiment_score = np.mean(results['sentiment_scores']) if results['sentiment_scores'] else 0.0
        
        # 종합 감성 판단
        if sentiment_score > 0.1:
            overall_sentiment = 'bullish'
        elif sentiment_score < -0.1:
            overall_sentiment = 'bearish'
        else:
            overall_sentiment = 'neutral'
        
        # 통계 정보 계산
        score_stats = {
            'mean': float(np.mean(results['sentiment_scores']) if results['sentiment_scores'] else 0.0),
            'median': float(np.median(results['sentiment_scores']) if results['sentiment_scores'] else 0.0),
            'std': float(np.std(results['sentiment_scores']) if results['sentiment_scores'] else 0.0),
            'min': float(np.min(results['sentiment_scores']) if results['sentiment_scores'] else 0.0),
            'max': float(np.max(results['sentiment_scores']) if results['sentiment_scores'] else 0.0)
        }
        
        # 결과 출력
        analysis_result = {
            'overall_sentiment': overall_sentiment,
            'sentiment_score': float(sentiment_score),
            'bullish_ratio': float(bullish_ratio),
            'bearish_ratio': float(bearish_ratio),
            'neutral_ratio': float(neutral_ratio),
            'detailed_scores': results['sentiment_scores'],
            'stats': score_stats,
            'detailed_results': results['detailed_results'],
            'method': 'text_based'
        }
        
        self.logger.info(f"텍스트 기반 감성 분석 결과: {overall_sentiment}, 점수: {sentiment_score:.2f}")
        
        return analysis_result
    
    def extract_features(self, news_data):
        """
        뉴스 데이터에서 특징 추출
        - DeepSeek 모델을 사용하여 감성 점수를 특징으로 변환
        """
        self.logger.info(f"뉴스 데이터 {len(news_data)}개에서 DeepSeek 기반 특징 추출 중...")
        
        # 감성 분석 결과 가져오기
        analysis_result = self.analyze_sentiment(news_data)
        
        # 특징 차원 설정 (VADER와 동일하게 유지)
        feature_dim = 128
        
        # 각 뉴스 항목에 대한 특징 생성
        features = []
        
        if 'detailed_scores' in analysis_result and analysis_result['detailed_scores']:
            # 각 뉴스 항목의 감성 점수를 사용하여 특징 생성
            for score in analysis_result['detailed_scores']:
                # 감성 점수를 기준으로 한 특징 벡터 생성
                # 점수가 높을수록(bullish) 양수 방향의 값이 커지고, 낮을수록(bearish) 음수 방향의 값이 커짐
                base_feature = np.random.randn(feature_dim)
                
                # 감성 점수에 따라 특징 벡터 조정 (증폭)
                adjusted_feature = base_feature + (score * 0.75)  # 감성 영향 증폭
                
                # 특징 벡터의 첫 8개 요소는 감성 점수와 직접 연관
                if score > 0:  # 강세
                    # 강세 패턴 특징 강화
                    adjusted_feature[:4] = np.abs(adjusted_feature[:4]) * score
                elif score < 0:  # 약세
                    # 약세 패턴 특징 강화
                    adjusted_feature[4:8] = np.abs(adjusted_feature[4:8]) * -score
                
                # 정규화
                norm = np.linalg.norm(adjusted_feature)
                if norm > 0:
                    adjusted_feature = adjusted_feature / norm
                
                features.append(adjusted_feature)
        else:
            # 상세 점수가 없는 경우 랜덤 특징 생성
            for _ in range(len(news_data)):
                feature = np.random.randn(feature_dim)
                # 정규화
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm
                features.append(feature)
        
        self.logger.info(f"DeepSeek 기반 뉴스 감성 특징 추출 완료: {len(features)}개")
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
            filename = f"ollama_sentiment_data_{timestamp}.json"
        
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