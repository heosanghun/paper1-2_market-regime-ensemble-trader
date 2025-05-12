#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
고급 멀티모달 트레이딩 시스템 실행 스크립트
- 멀티모달 데이터(차트 이미지, 뉴스 감성) 기반 트레이딩
- 강화학습과 앙상블 방법론 조합
"""

import os
import sys
import time
import torch
import logging
import colorama
from datetime import datetime
from tqdm import tqdm

from advanced_multimodal_trader import AdvancedMultimodalTrader
from candlestick_analyzer import CandlestickAnalyzer
from sentiment_analyzer import SentimentAnalyzer

# 컬러 출력 초기화
colorama.init()

# 색상 코드 정의
GREEN = colorama.Fore.GREEN
YELLOW = colorama.Fore.YELLOW
RED = colorama.Fore.RED
BLUE = colorama.Fore.BLUE
MAGENTA = colorama.Fore.MAGENTA
CYAN = colorama.Fore.CYAN
RESET = colorama.Fore.RESET
BRIGHT = colorama.Style.BRIGHT
DIM = colorama.Style.DIM

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AdvancedTraderRunner')

def print_status(message, color=GREEN, is_title=False):
    """상태 메시지 출력 함수"""
    timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    
    if is_title:
        print("\n" + "="*80)
        print(f"{BRIGHT}{color}{timestamp} {message}{RESET}")
        print("="*80 + "\n")
    else:
        print(f"{color}{timestamp} {message}{RESET}")

def format_time(seconds):
    """시간 형식화 함수"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{int(hours)}시간 {int(minutes)}분 {seconds:.2f}초"
    elif minutes > 0:
        return f"{int(minutes)}분 {seconds:.2f}초"
    else:
        return f"{seconds:.2f}초"

def main():
    # 시작 시간 기록
    start_time = time.time()
    print_status("고급 멀티모달 트레이딩 시뮬레이션 시작", MAGENTA, True)
    
    # GPU 확인 및 사용 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_status(f"사용 가능 디바이스: {device}", CYAN)
    
    # 실행 경로 확인
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    print_status(f"현재 작업 디렉토리: {os.getcwd()}", BLUE)
    
    # 결과 디렉토리 생성
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 실행 시간을 파일명에 포함
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_results_dir = os.path.join(results_dir, f'run_{timestamp}')
    os.makedirs(run_results_dir, exist_ok=True)
    print_status(f"결과 저장 경로: {run_results_dir}", BLUE)
    
    # 설정 정의
    config = {
        'data': {
            'chart_dir': r'D:\drl-candlesticks-trader-main1\paper1\data\chart',  # 차트 데이터 경로
            'news_file': r'D:\drl-candlesticks-trader-main1\paper1\data\news\cryptonews_2021-10-12_2023-12-19.csv',  # 뉴스 데이터 파일
            'timeframes': ['1d', '4h']  # 1d와 4h 시간프레임 모두 사용
        },
        'output': {
            'save_dir': run_results_dir  # 결과 저장 경로
        },
        # 고급 기술 설정 추가
        'fusion': {
            'type': 'attention'  # 'attention' 또는 'transformer'
        },
        'rl': {
            'use_rl': True,  # 강화학습 사용 여부
            'learning_rate': 0.0003,
            'gamma': 0.99
        },
        'ensemble': {
            'use_ensemble': True,  # 앙상블 방법론 사용 여부
            'timeframe_ensemble_method': 'weighted_average',
            'model_ensemble_method': 'weighted_average',
            'dynamic_weights': True
        },
        'trading': {
            'initial_balance': 10000.0,  # 초기 자금
            'transaction_fee': 0.001,    # 거래 수수료 (0.1%)
            'risk_free_rate': 0.02       # 연간 무위험 수익률 (2%)
        }
    }
    
    try:
        # 전체 진행 단계 정의
        steps = [
            '캔들스틱 분석기 초기화', 
            '감성 분석기 초기화', 
            '고급 트레이더 초기화', 
            '트레이딩 시뮬레이션 실행', 
            '성능 평가', 
            '결과 저장', 
            '논문 제출용 결과 생성'
        ]
        
        # 진행 상황 표시
        overall_progress = tqdm(
            total=len(steps), 
            desc=f"{BRIGHT}{CYAN}[전체 진행률]{RESET}", 
            position=0, 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # 각 단계별 실행
        for i, step in enumerate(steps):
            step_start_time = time.time()
            step_desc = f"{GREEN}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {step} 중...{RESET} ({i+1}/{len(steps)})"
            print(f"\n{step_desc}")
            
            # 단계별 진행 및 실행
            if step == '캔들스틱 분석기 초기화':
                print_status("캔들스틱 차트 데이터 로딩 중...", YELLOW)
                candlestick_analyzer = CandlestickAnalyzer(config)
                print_status("캔들스틱 패턴 인식 모델 초기화 완료", GREEN)
            
            elif step == '감성 분석기 초기화':
                print_status("뉴스 감성 데이터 로딩 중...", YELLOW)
                sentiment_analyzer = SentimentAnalyzer(config)
                print_status("감성 분석 모델 초기화 완료", GREEN)
            
            elif step == '고급 트레이더 초기화':
                print_status("고급 멀티모달 트레이더 초기화 중...", YELLOW)
                print_status("- 앙상블 모델 구성 중...", DIM + YELLOW)
                print_status("- 강화학습 모델 초기화 중...", DIM + YELLOW)
                trader = AdvancedMultimodalTrader(candlestick_analyzer, sentiment_analyzer, config)
                print_status("고급 멀티모달 트레이더 초기화 완료", GREEN)
            
            elif step == '트레이딩 시뮬레이션 실행':
                print_status("트레이딩 시뮬레이션 실행 중...", YELLOW)
                step_progress = tqdm(total=100, desc="[시뮬레이션 진행률]", position=1)
                
                # 시뮬레이션 업데이트 콜백 함수
                def update_progress(percent, info=""):
                    step_progress.n = int(percent)
                    step_progress.set_postfix_str(info)
                    step_progress.refresh()
                
                # 콜백 함수 등록 및 실행
                trader.set_progress_callback(update_progress)
                trader.run()
                step_progress.close()
                print_status("트레이딩 시뮬레이션 실행 완료", GREEN)
            
            elif step == '성능 평가':
                print_status("성능 평가 중...", YELLOW)
                trader.evaluate_performance()
                print_status("성능 평가 완료", GREEN)
            
            elif step == '결과 저장':
                print_status("결과 저장 중...", YELLOW)
                trader.save_results()
                print_status("결과 저장 완료", GREEN)
            
            elif step == '논문 제출용 결과 생성':
                print_status("논문 제출용 성능 지표 및 시각화 생성 중...", YELLOW)
                subprogress = tqdm(total=5, desc="[보고서 생성 진행률]", position=1)
                
                # 논문용 보고서 생성 (성능 지표 분석)
                subprogress.update(1); subprogress.refresh()
                print_status("- 기본 성능 지표 계산 중...", DIM + YELLOW)
                
                # Expert Systems with Applications 논문 형식 지표 추가
                subprogress.update(1); subprogress.refresh()
                print_status("- Expert Systems with Applications 논문 형식 지표 생성 중...", DIM + YELLOW)
                trader.generate_expert_systems_metrics()
                
                # 시각화 자료 생성
                subprogress.update(1); subprogress.refresh()
                print_status("- 시각화 자료 생성 중...", DIM + YELLOW)
                
                # 최종 보고서 생성
                subprogress.update(1); subprogress.refresh()
                print_status("- 최종 보고서 생성 중...", DIM + YELLOW)
                trader.paper_report_for_submission()
                
                # 완료
                subprogress.update(1); subprogress.refresh()
                subprogress.close()
                print_status("논문 제출용 결과 생성 완료", GREEN)
            
            # 각 단계 완료 시간 측정
            step_elapsed_time = time.time() - step_start_time
            print_status(f"{step} 완료 (소요 시간: {format_time(step_elapsed_time)})", CYAN)
            
            # 전체 진행률 업데이트
            overall_progress.update(1)
        
        # 전체 실행 시간 계산
        elapsed_time = time.time() - start_time
        
        print_status("고급 멀티모달 트레이딩 시뮬레이션 완료!", MAGENTA, True)
        print_status(f"총 소요 시간: {format_time(elapsed_time)}", CYAN)
        print_status(f"결과가 다음 경로에 저장되었습니다: {run_results_dir}", BLUE)
        
        overall_progress.close()
        return 0
    
    except Exception as e:
        logger.error(f"시뮬레이션 중 오류 발생: {str(e)}", exc_info=True)
        
        # 오류 발생 시간 기록
        elapsed_time = time.time() - start_time
        
        print_status("오류 발생으로 시뮬레이션 중단!", RED, True)
        print_status(f"실행 시간: {format_time(elapsed_time)}", YELLOW)
        print_status(f"오류 내용: {str(e)}", RED)
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 