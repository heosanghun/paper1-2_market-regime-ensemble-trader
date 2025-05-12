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
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

class DeepSeekNewsAnalyzer:
    """
    Ollama API를 사용하여 DeepSeek-R1(32B) 모델 기반 뉴스 분석을 수행하는 클래스
    - Chain-of-Thought 추론을 통한 시장 영향 분석
    - 암호화폐 뉴스의 시장 영향도 (-1~1) 분석
    - 768차원 특징 벡터 생성
    """
    def __init__(self, config=None):
        """
        DeepSeek 뉴스 분석기 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        # 설정 저장
        self.config = config or {}
        
        # 설정에서 결과 저장 경로 가져오기
        output_config = self.config.get('output', {})
        self.results_dir = output_config.get('save_dir', 'results/paper1_deepseek')
        
        # 디렉토리 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 로거 설정
        self.logger = logging.getLogger('DeepSeekNewsAnalyzer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        # Ollama API URL
        self.ollama_url = self.config.get('ollama_url', 'http://localhost:11434/api/generate')
        
        # DeepSeek 모델명 - 설정에서 모델명을 가져오거나 기본값 사용
        self.model_name = self.config.get('model_name', 'deepseek-llm:latest')
        
        # 재시도 횟수
        self.max_retries = self.config.get('max_retries', 3)
        
        # API 호출 간격 (초)
        self.api_call_interval = self.config.get('api_call_interval', 0.5)
        
        # 배치 크기 (동시 처리할 뉴스 항목 수)
        self.batch_size = self.config.get('batch_size', 1)
        
        # 오프라인 모드 - API가 실패하면 텍스트 기반 대체 로직 사용
        self.offline_mode = self.config.get('offline_mode', False)
        
        # 특징 벡터 크기
        self.feature_dim = self.config.get('feature_dim', 256)
        
        # 시장 영향 분석 프롬프트 템플릿 (Chain-of-Thought 추론 방식)
        self.cot_prompt_template = """아래 암호화폐 뉴스를 단계별로 분석하고 비트코인 가격에 미칠 영향을 평가해주세요:

뉴스: {news}

다음 단계로 진행하세요:
1) 이 뉴스의 핵심 내용과 주요 사실을 요약하세요.
2) 이 뉴스가 비트코인의 기반 기술, 채택, 규제, 또는 시장 심리에 어떤 영향을 미칠지 분석하세요.
3) 과거 유사한 뉴스가 시장에 미친 영향을 고려하세요.
4) 이 뉴스의 단기적(1-7일) 시장 영향을 -1(매우 부정적)에서 1(매우 긍정적) 사이의 점수로 평가하고, 그 이유를 설명하세요.

마지막으로, 아래 형식의 JSON으로만 응답해주세요:
{{
  "summary": "뉴스 핵심 내용 요약",
  "impact_analysis": "시장 영향 분석",
  "historical_context": "과거 유사 뉴스 영향",
  "market_impact_score": -1.0~1.0 사이의 값,
  "reasoning": "점수에 대한 근거 설명"
}}

설명이나 다른 텍스트는 포함하지 마세요. 오직 JSON만 응답하세요.
"""
        
        # 특징 추출 프롬프트 템플릿
        self.feature_extraction_prompt = """다음 암호화폐 뉴스의 주요 특징을 추출해주세요:

뉴스: {news}

다음 항목들을 분석하여 JSON 형식으로 추출해주세요:
1) 언급된 암호화폐 (비트코인, 이더리움 등)
2) 언급된 기관 또는 기업
3) 뉴스 유형 (규제, 채택, 기술, 시장 등)
4) 감성 극성 (매우 부정, 부정, 중립, 긍정, 매우 긍정)
5) 시장 영향 범위 (글로벌, 지역, 국가, 특정 부문)
6) 뉴스의 확실성/불확실성 정도 (확실, 추측, 불확실)
7) 시장에 미칠 시간적 영향 (즉각, 단기, 중기, 장기)

JSON 형식으로만 응답해주세요:
{{
  "cryptocurrencies": ["비트코인", ...],
  "entities": ["기관/기업명", ...],
  "news_type": "유형",
  "sentiment": "감성 극성",
  "impact_scope": "영향 범위",
  "certainty": "확실성 정도",
  "time_horizon": "시간적 영향"
}}
"""
        
        # Ollama 서버 연결 확인
        self._check_server_connection()
        
        self.logger.info(f"DeepSeek 뉴스 분석기 초기화 완료 (모델: {self.model_name}, {'오프라인 모드' if self.offline_mode else '온라인 모드'})")
    
    def _check_server_connection(self):
        """Ollama 서버 연결 확인 및 모델 확인"""
        try:
            # 간단한 API 호출로 서버 연결 확인
            url = "http://localhost:11434/api/tags"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                # 사용 가능한 모델 목록 가져오기
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                self.logger.info(f"Ollama 서버 연결 성공, 사용 가능한 모델: {len(models)}개")
                
                # DeepSeek 모델 우선 찾기
                deepseek_models = [m for m in model_names if 'deepseek' in m.lower()]
                if deepseek_models:
                    # DeepSeek 모델이 있으면 선택
                    self.model_name = deepseek_models[0]
                    self.logger.info(f"DeepSeek 모델 '{self.model_name}'을(를) 사용합니다.")
                elif 'llama3' in model_names or 'llama3:latest' in model_names:
                    # DeepSeek이 없으면 Llama3 사용
                    self.model_name = 'llama3:latest' if 'llama3:latest' in model_names else 'llama3'
                    self.logger.info(f"DeepSeek 모델이 없어 '{self.model_name}'을(를) 사용합니다.")
                elif len(model_names) > 0:
                    # 아무 모델이나 사용
                    self.model_name = model_names[0]
                    self.logger.info(f"선호 모델이 없어 '{self.model_name}'을(를) 사용합니다.")
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
        """
        뉴스 데이터 가져오기
        
        Args:
            symbol: 암호화폐 심볼
            days_back: 몇 일 전까지의 뉴스를 가져올지
            
        Returns:
            DataFrame: 뉴스 데이터
        """
        try:
            # 실제 구현에서는 API에서 데이터를 가져오지만, 예시에서는 샘플 데이터 사용
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # 샘플 제목과 내용
            sample_news = [
                {
                    "title": "Bitcoin ETF Approval Expected Soon, SEC Chairman Hints",
                    "content": "The SEC chairman hinted that the approval of Bitcoin ETFs might be coming sooner than expected, potentially opening the cryptocurrency to more institutional investors."
                },
                {
                    "title": "Major Bank Announces Bitcoin Custody Services for Institutional Clients",
                    "content": "One of the world's largest banks announced today that it will begin offering Bitcoin custody services to its institutional clients, marking a significant step in cryptocurrency adoption by traditional financial institutions."
                },
                {
                    "title": "New Regulations Threaten Crypto Exchanges in Key Asian Markets",
                    "content": "Regulatory bodies in several Asian countries announced stricter oversight of cryptocurrency exchanges, raising concerns about the future of crypto trading in these key markets."
                },
                {
                    "title": "Bitcoin Mining Difficulty Hits All-Time High",
                    "content": "Bitcoin's mining difficulty has reached a new all-time high, indicating increased competition among miners and potentially greater network security."
                },
                {
                    "title": "Leading Tech Company Adds Bitcoin to Corporate Treasury",
                    "content": "A Fortune 500 technology company has announced the addition of Bitcoin to its corporate treasury, following the trend set by other major companies investing in cryptocurrency as an inflation hedge."
                },
                {
                    "title": "Bitcoin Lightning Network Capacity Grows 200% in Six Months",
                    "content": "The Bitcoin Lightning Network has seen its capacity grow by 200% over the past six months, suggesting increased adoption of Bitcoin's layer-2 scaling solution."
                },
                {
                    "title": "Crypto Market Faces Uncertainty as Global Economic Concerns Mount",
                    "content": "Cryptocurrency markets are experiencing increased volatility as global economic concerns, including inflation and potential recession, continue to influence investor sentiment."
                }
            ]
            
            # 날짜 범위 생성
            date_range = pd.date_range(start=start_date, end=end_date, periods=len(sample_news))
            
            # 데이터 프레임 생성
            news_data = []
            for i, date in enumerate(date_range):
                news = sample_news[i % len(sample_news)]
                news_data.append({
                    'date': date,
                    'title': news['title'],
                    'content': news['content'],
                    'source': f"CryptoNews Source {i+1}"
                })
            
            df = pd.DataFrame(news_data)
            
            self.logger.info(f"샘플 뉴스 데이터 생성 완료: {len(df)} 항목")
            
            # 결과 저장
            csv_path = os.path.join(self.results_dir, f"{symbol}_news_data.csv")
            df.to_csv(csv_path, index=False)
            
            return df
        
        except Exception as e:
            self.logger.error(f"뉴스 데이터 가져오기 오류: {str(e)}")
            return pd.DataFrame()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call_ollama_api(self, prompt):
        """
        Ollama API를 호출하여 DeepSeek 모델의 응답 받기
        
        Args:
            prompt: 프롬프트
            
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
                    "temperature": 0.01,  # 낮은 온도로 일관된 응답 유도
                    "num_predict": 2048,  # 충분한 응답 길이 허용
                    "top_p": 0.1,
                    "top_k": 10
                }
            }
            
            # API 호출
            response = requests.post(self.ollama_url, json=data, timeout=30)
            
            # 오류 체크
            if response.status_code != 200:
                self.logger.error(f"Ollama API 오류: {response.status_code} - {response.text}")
                raise Exception(f"API 호출 실패: 상태 코드 {response.status_code}")
            
            # 응답 파싱
            result = response.json()
            response_text = result.get('response', '')
            
            self.logger.info(f"API 응답 수신 (길이: {len(response_text)} 문자)")
            
            # JSON 추출
            extracted_json = self._extract_json_from_text(response_text)
            
            if not extracted_json:
                self.logger.warning("JSON 응답을 추출할 수 없습니다. 기본값 반환")
                return self._generate_default_response()
            
            return extracted_json
            
        except Exception as e:
            self.logger.error(f"API 호출 오류: {str(e)}")
            return self._generate_default_response()
    
    def _extract_json_from_text(self, text):
        """
        텍스트에서 JSON 부분 추출
        
        Args:
            text: 응답 텍스트
            
        Returns:
            dict: 추출된 JSON 객체
        """
        try:
            # 정규식으로 JSON 찾기
            json_pattern = r'({[\s\S]*})'
            matches = re.findall(json_pattern, text)
            
            if matches:
                # 첫 번째 매치 시도
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
            
            # JSON이 깨끗하게 추출되지 않은 경우, 전체 텍스트를 JSON으로 파싱 시도
            return json.loads(text)
            
        except Exception as e:
            self.logger.error(f"JSON 추출 오류: {str(e)}")
            return None
    
    def _generate_default_response(self):
        """
        기본 응답 생성 (API 호출 실패 시)
        
        Returns:
            dict: 기본 응답
        """
        return {
            "summary": "뉴스 요약을 생성할 수 없습니다.",
            "impact_analysis": "영향 분석을 수행할 수 없습니다.",
            "historical_context": "과거 맥락을 분석할 수 없습니다.",
            "market_impact_score": 0.0,
            "reasoning": "API 호출 실패로 분석을 수행할 수 없습니다."
        }
    
    def analyze_news(self, news_data):
        """
        뉴스 데이터 분석
        
        Args:
            news_data: 뉴스 데이터프레임
            
        Returns:
            DataFrame: 분석 결과
        """
        if news_data.empty:
            self.logger.warning("분석할 뉴스 데이터가 없습니다.")
            return pd.DataFrame()
        
        self.logger.info(f"뉴스 분석 시작: {len(news_data)} 항목")
        
        # 오프라인 모드에서는 간단한 대체 분석 수행
        if self.offline_mode:
            return self._analyze_news_offline(news_data)
        
        # 결과 저장용 리스트
        results = []
        
        # 프로그레스 바 표시
        for i, row in tqdm(news_data.iterrows(), total=len(news_data), desc="뉴스 분석"):
            # 제목과 내용 결합
            news_text = f"{row['title']}. {row['content']}"
            
            # 분석 프롬프트 생성
            prompt = self.cot_prompt_template.format(news=news_text)
            
            # API 호출
            analysis = self.call_ollama_api(prompt)
            
            # 결과 저장
            result = {
                'date': row['date'],
                'title': row['title'],
                'content': row['content'],
                'source': row.get('source', ''),
                'summary': analysis.get('summary', ''),
                'impact_analysis': analysis.get('impact_analysis', ''),
                'historical_context': analysis.get('historical_context', ''),
                'market_impact_score': analysis.get('market_impact_score', 0.0),
                'reasoning': analysis.get('reasoning', '')
            }
            
            results.append(result)
            
            # API 요청 간 간격 두기
            time.sleep(self.api_call_interval)
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame(results)
        
        # 결과 저장
        csv_path = os.path.join(self.results_dir, "news_analysis_results.csv")
        result_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"뉴스 분석 완료: {len(result_df)} 항목, 결과 저장: {csv_path}")
        
        return result_df
    
    def extract_features(self, news_data):
        """
        뉴스에서 특징 벡터 추출
        
        Args:
            news_data: 뉴스 데이터프레임 또는 분석 결과 데이터프레임
            
        Returns:
            특징 벡터 텐서 [배치, 특징_차원]
        """
        if news_data.empty:
            self.logger.warning("특징 추출을 위한 뉴스 데이터가 없습니다.")
            return torch.zeros((1, self.feature_dim))
        
        # news_data가 이미 분석 결과인지 확인
        is_analyzed = 'market_impact_score' in news_data.columns
        
        if not is_analyzed:
            # 분석 결과가 없으면 분석 먼저 수행
            analyzed_data = self.analyze_news(news_data)
        else:
            analyzed_data = news_data
        
        # 특징 추출을 위한 데이터 준비
        features = []
        
        for i, row in analyzed_data.iterrows():
            # 1. 시장 영향 점수 (가장 중요한 특징) - 스케일링해서 사용
            market_score = float(row.get('market_impact_score', 0.0))
            
            # 2. 텍스트 기반 특징 추출 (키워드 기반의 간단한 방법)
            # 실제 구현에서는 임베딩 모델을 사용하는 것이 좋지만, 여기서는 간단한 방법 사용
            feature_vec = np.zeros(self.feature_dim)
            
            # 기본 차원에 시장 영향 점수 할당
            feature_vec[0] = market_score
            
            # 요약 정보 기반 특징
            summary = row.get('summary', '')
            if 'adoption' in summary.lower() or '채택' in summary:
                feature_vec[1] = 1.0
            if 'regulation' in summary.lower() or '규제' in summary:
                feature_vec[2] = 1.0
            if 'technology' in summary.lower() or '기술' in summary:
                feature_vec[3] = 1.0
            if 'institution' in summary.lower() or '기관' in summary:
                feature_vec[4] = 1.0
            
            # 영향 분석 기반 특징
            impact = row.get('impact_analysis', '')
            if 'positive' in impact.lower() or '긍정적' in impact:
                feature_vec[5] = 1.0
            if 'negative' in impact.lower() or '부정적' in impact:
                feature_vec[6] = -1.0
            if 'short-term' in impact.lower() or '단기' in impact:
                feature_vec[7] = 1.0
            if 'long-term' in impact.lower() or '장기' in impact:
                feature_vec[8] = 1.0
            
            # 남은 차원은 정규 분포의 랜덤 값으로 채움 (실제 모델에서는 더 의미 있는 값 사용)
            feature_vec[9:] = np.random.normal(0, 0.1, self.feature_dim - 9)
            
            # 벡터 정규화
            if np.linalg.norm(feature_vec) > 0:
                feature_vec = feature_vec / np.linalg.norm(feature_vec)
            
            features.append(feature_vec)
        
        # 모든 뉴스의 특징을 평균하여 최종 특징 벡터 생성
        if features:
            final_features = np.mean(features, axis=0)
        else:
            final_features = np.zeros(self.feature_dim)
        
        # 텐서로 변환
        feature_tensor = torch.FloatTensor(final_features).unsqueeze(0)
        
        self.logger.info(f"뉴스 특징 추출 완료: 형태={feature_tensor.shape}")
        
        return feature_tensor
    
    def _analyze_news_offline(self, news_data):
        """
        오프라인 모드에서 간단한 뉴스 분석 수행
        
        Args:
            news_data: 뉴스 데이터프레임
            
        Returns:
            DataFrame: 분석 결과
        """
        self.logger.info(f"오프라인 모드에서 뉴스 분석 시작: {len(news_data)} 항목")
        
        # 긍정/부정 키워드 사전 (간단한 규칙 기반 분석)
        positive_keywords = [
            'adoption', 'institutional', 'growth', 'bullish', 'rally', 'surge',
            'increase', 'gain', 'positive', 'approval', 'launch', 'investment',
            'partnership', 'innovation', '채택', '기관', '성장', '상승', '긍정'
        ]
        
        negative_keywords = [
            'regulation', 'ban', 'bearish', 'crash', 'decline', 'loss',
            'concern', 'risk', 'hack', 'scam', 'fraud', 'investigation',
            'crackdown', 'volatility', '규제', '금지', '하락', '손실', '우려'
        ]
        
        # 결과 저장용 리스트
        results = []
        
        for i, row in news_data.iterrows():
            combined_text = f"{row['title']} {row['content']}".lower()
            
            # 긍정/부정 키워드 카운트
            positive_count = sum(1 for keyword in positive_keywords if keyword.lower() in combined_text)
            negative_count = sum(1 for keyword in negative_keywords if keyword.lower() in combined_text)
            
            # 시장 영향 점수 계산 (-1.0 ~ 1.0)
            total_count = positive_count + negative_count
            if total_count == 0:
                impact_score = 0.0
            else:
                impact_score = (positive_count - negative_count) / total_count
                impact_score = max(-1.0, min(1.0, impact_score))  # 범위 제한
            
            # 간단한 요약 생성
            summary = f"이 뉴스는 암호화폐 시장에 관한 내용입니다."
            
            # 결과 저장
            result = {
                'date': row['date'],
                'title': row['title'],
                'content': row['content'],
                'source': row.get('source', ''),
                'summary': summary,
                'impact_analysis': f"이 뉴스는 시장에 {'긍정적' if impact_score > 0 else '부정적' if impact_score < 0 else '중립적'} 영향을 줄 것으로 분석됩니다.",
                'historical_context': "오프라인 모드에서는 과거 맥락 분석이 제한적입니다.",
                'market_impact_score': impact_score,
                'reasoning': f"긍정 키워드: {positive_count}개, 부정 키워드: {negative_count}개 발견"
            }
            
            results.append(result)
        
        # 결과 데이터프레임 생성
        result_df = pd.DataFrame(results)
        
        # 결과 저장
        csv_path = os.path.join(self.results_dir, "news_analysis_results_offline.csv")
        result_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"오프라인 뉴스 분석 완료: {len(result_df)} 항목, 결과 저장: {csv_path}")
        
        return result_df

# 테스트 함수
def test_deepseek_news_analyzer():
    # 설정
    config = {
        'model_name': 'deepseek-llm:latest',  # 또는 사용 가능한 다른 모델
        'offline_mode': False,  # Ollama 서버가 없으면 True로 설정
        'output': {
            'save_dir': 'results/deepseek_test'
        }
    }
    
    # 분석기 초기화
    analyzer = DeepSeekNewsAnalyzer(config)
    
    # 뉴스 데이터 가져오기
    news_data = analyzer.fetch_news(days_back=3)
    
    # 뉴스 분석
    analysis_results = analyzer.analyze_news(news_data)
    
    # 결과 출력
    if not analysis_results.empty:
        for i, row in analysis_results.iterrows():
            print(f"\n==== 뉴스 {i+1} ====")
            print(f"제목: {row['title']}")
            print(f"요약: {row['summary']}")
            print(f"시장 영향 점수: {row['market_impact_score']}")
            print(f"근거: {row['reasoning'][:100]}...")
    
    # 특징 추출
    features = analyzer.extract_features(analysis_results)
    print(f"\n특징 벡터 크기: {features.shape}")
    print(f"특징 벡터 일부: {features[0, :10]}")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    test_deepseek_news_analyzer() 