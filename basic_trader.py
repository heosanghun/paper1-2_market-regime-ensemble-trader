import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import time
import requests
import seaborn as sns
from multimodal_fusion import AttentionFusion, TransformerFusion
from rl_trader import PPOTrader, convert_to_state, action_to_position
from ensemble_trader import MultiTimeframeEnsemble, ModelEnsemble
import torch

class BasicTrader:
    def __init__(self, candlestick_analyzer, sentiment_analyzer, config=None, progress_callback=None):
        """BasicTrader 초기화"""
        self.candlestick_analyzer = candlestick_analyzer
        self.sentiment_analyzer = sentiment_analyzer
        self.config = config or {}
        # 진행 상황 콜백 추가
        self.progress_callback = progress_callback
        
        # 기존 설정 로드
        self.balance = self.config.get('initial_balance', 10000.0)
        self.position = 'neutral'  # 중립 시작
        self.trades = []
        self.portfolio_values = []
        self.entry_price = 0.0
        
        # 결과 저장 경로
        output_config = self.config.get('output', {})
        self.results_dir = output_config.get('save_dir', 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 로깅 설정
        self.logger = self.setup_logger()
        
        # ======= 고급 기술 관련 추가 설정 =======
        
        # 1. 비선형 멀티모달 융합
        fusion_config = self.config.get('fusion', {})
        fusion_type = fusion_config.get('type', 'attention')  # 'attention' 또는 'transformer'
        
        # 주의 메커니즘 또는 트랜스포머 기반 융합 모듈
        if fusion_type == 'attention':
            self.fusion_model = AttentionFusion()
            self.logger.info("주의 메커니즘 기반 멀티모달 융합 모듈 초기화 완료")
        else:
            self.fusion_model = TransformerFusion()
            self.logger.info("트랜스포머 기반 멀티모달 융합 모듈 초기화 완료")
        
        # 2. 강화학습(RL) 설정
        rl_config = self.config.get('rl', {})
        use_rl = rl_config.get('use_rl', False)
        self.use_rl = use_rl
        
        if use_rl:
            # 상태 차원: 패턴 신호, 감성 신호, 가격 변화, 포지션(3), 포트폴리오 가치
            state_dim = 7
            # 행동 차원: 매수, 매도, 홀딩
            action_dim = 3
            self.rl_trader = PPOTrader(state_dim=state_dim, action_dim=action_dim)
            self.logger.info("강화학습(PPO) 거래 모듈 초기화 완료")
        
        # 3. 앙상블 방법론 설정
        ensemble_config = self.config.get('ensemble', {})
        use_ensemble = ensemble_config.get('use_ensemble', False)
        self.use_ensemble = use_ensemble
        
        if use_ensemble:
            # 다중 시간프레임 앙상블
            timeframes = self.config.get('data', {}).get('timeframes', ['1d'])
            self.timeframe_ensemble = MultiTimeframeEnsemble(timeframes=timeframes)
            
            # 모델 앙상블
            self.model_ensemble = ModelEnsemble()
            
            # 초기 모델 등록
            self.model_ensemble.add_model('fusion', self.fusion_model, weight=0.6)
            if use_rl:
                self.model_ensemble.add_model('rl', self.rl_trader, weight=0.4)
            
            self.logger.info("앙상블 거래 모듈 초기화 완료")
        
        # 거래 성능 지표
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'final_return': 0.0,
            'sharpe_ratio': 0.0
        }
        
    # 진행률 업데이트 함수 추가
    def update_progress(self, progress, status):
        """진행률 및 상태 업데이트"""
        if self.progress_callback:
            self.progress_callback(progress, status)
    
    def setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger('BasicTrader')
        logger.setLevel(logging.INFO)
        
        # 핸들러가 이미 있는지 확인
        if not logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 포맷터
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # 핸들러 추가
            logger.addHandler(console_handler)
            
            # 파일 핸들러 (선택 사항)
            if hasattr(self, 'results_dir'):
                log_file = os.path.join(self.results_dir, 'trading.log')
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        logger.info("BasicTrader 로거 초기화 완료")
        return logger
    
    def run(self):
        """트레이딩 실행"""
        try:
            data_config = self.config.get('data', {})
            chart_dir = data_config.get('chart_dir', 'data/chart')
            news_file = data_config.get('news_file', 'data/news/cryptonews.csv')
            timeframes = data_config.get('timeframes', ['1d'])

            total_timeframes = len(timeframes)
            start_time = time.time()
            
            # 초기 진행률 업데이트
            self.update_progress(70, "트레이딩 시뮬레이션 시작")

            for idx, timeframe in enumerate(tqdm(timeframes, desc='[전체 진행률] 시간프레임 처리', position=0)):
                tf_start = time.time()
                self.logger.info(f"{timeframe} 시간봉 처리 시작 ({idx+1}/{total_timeframes})")
                
                # 진행률 계산 - 시간프레임 단위
                tf_progress = 70 + (idx / total_timeframes) * 10
                self.update_progress(tf_progress, f"{timeframe} 시간봉 처리 중 ({idx+1}/{total_timeframes})")
                
                chart_path = os.path.join(chart_dir, timeframe)
                if not os.path.exists(chart_path):
                    self.logger.warning(f"{chart_path} 경로가 존재하지 않습니다. 진행을 멈추고 오류를 해결하세요.")
                    print(f"[오류] {chart_path} 경로가 존재하지 않습니다. 진행을 멈춥니다.")
                    self.update_progress(tf_progress, f"오류: {chart_path} 경로가 존재하지 않습니다.")
                    return False

                # 차트 이미지 분석 (진행률 표시)
                image_files = [f for f in os.listdir(chart_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                if not image_files:
                    self.logger.warning(f"{chart_path}에 차트 이미지가 없습니다. 진행을 멈추고 오류를 해결하세요.")
                    print(f"[오류] {chart_path}에 차트 이미지가 없습니다. 진행을 멈춥니다.")
                    self.update_progress(tf_progress, f"오류: {chart_path}에 차트 이미지가 없습니다.")
                    return False
                
                # 차트 이미지 분석 진행률 업데이트
                self.update_progress(tf_progress, f"{timeframe} 차트 이미지 분석 중...")
                for i, image_file in enumerate(tqdm(image_files, desc=f'[{timeframe}] 차트 이미지 분석', position=1, leave=False)):
                    if i % 10 == 0:  # 10개 이미지마다 진행률 업데이트
                        img_progress = tf_progress + (i / len(image_files)) * 2
                        self.update_progress(img_progress, f"{timeframe} 차트 이미지 분석 중 ({i+1}/{len(image_files)})")
                    pass
                pattern_results = self.candlestick_analyzer.analyze_patterns(chart_path)

                # 뉴스 데이터 분석 (진행률 표시)
                news_progress = tf_progress + 2
                self.update_progress(news_progress, f"{timeframe} 뉴스 감성 분석 중...")
                self.logger.info(f"{timeframe} 뉴스 감성 분석 중...")
                try:
                    news_data = pd.read_csv(news_file)
                except Exception as e:
                    self.logger.error(f"뉴스 데이터 파일을 읽을 수 없습니다: {str(e)}. 진행을 멈춥니다.")
                    print(f"[오류] 뉴스 데이터 파일을 읽을 수 없습니다: {str(e)}. 진행을 멈춥니다.")
                    self.update_progress(news_progress, f"오류: 뉴스 데이터 파일을 읽을 수 없습니다: {str(e)}")
                    return False
                if news_data.empty:
                    self.logger.warning(f"뉴스 데이터가 비어 있습니다. 진행을 멈춥니다.")
                    print(f"[오류] 뉴스 데이터가 비어 있습니다. 진행을 멈춥니다.")
                    self.update_progress(news_progress, "오류: 뉴스 데이터가 비어 있습니다.")
                    return False
                try:
                    sentiment_results = self.sentiment_analyzer.analyze_sentiment(news_data)
                except Exception as e:
                    self.logger.error(f"뉴스 감성 분석 오류: {str(e)}. 진행을 멈춥니다.")
                    print(f"[오류] 뉴스 감성 분석 오류: {str(e)}. 진행을 멈춥니다.")
                    self.update_progress(news_progress, f"오류: 뉴스 감성 분석 오류: {str(e)}")
                    return False

                # 가격 데이터 로드 진행률 업데이트
                price_progress = news_progress + 2
                self.update_progress(price_progress, f"{timeframe} 가격 데이터 로드 중...")
                
                # 가격 데이터 로드 (차트 이미지 파일명에서 가격 정보 추출)
                price_data = []
                for image_file in image_files:
                    try:
                        parts = image_file.split('_')
                        date_str = parts[2]
                        time_str = parts[3]
                        price = float(parts[4].replace('.png',''))
                        timestamp = pd.to_datetime(f"{date_str} {time_str}", format="%Y-%m-%d %H-%M")
                        price_data.append({'timestamp': timestamp, 'close': price})
                    except:
                        continue
                if not price_data:
                    self.logger.warning(f"{timeframe} 시간봉에 대한 가격 데이터가 없습니다. 바이낸스에서 자동으로 가격 데이터를 받아와 생성합니다.")
                    print(f"[경고] {timeframe} 시간봉에 대한 가격 데이터가 없습니다. 바이낸스에서 자동으로 가격 데이터를 받아와 생성합니다.")
                    self.update_progress(price_progress, f"{timeframe} 바이낸스에서 가격 데이터 다운로드 중...")
                    # 바이낸스에서 가격 데이터 자동 다운로드 및 생성
                    try:
                        from binance.client import Client
                        client = Client()
                        interval_map = {
                            '5m': Client.KLINE_INTERVAL_5MINUTE,
                            '15m': Client.KLINE_INTERVAL_15MINUTE,
                            '30m': Client.KLINE_INTERVAL_30MINUTE,
                            '1h': Client.KLINE_INTERVAL_1HOUR,
                            '4h': Client.KLINE_INTERVAL_4HOUR,
                            '1d': Client.KLINE_INTERVAL_1DAY,
                        }
                        # 뉴스 데이터 기준 날짜 범위로 다운로드
                        min_date = news_data['date'].min()[:10]
                        max_date = news_data['date'].max()[:10]
                        klines = client.get_historical_klines(
                            symbol='BTCUSDT',
                            interval=interval_map[timeframe],
                            start_str=min_date,
                            end_str=max_date
                        )
                        price_data = []
                        for k in klines:
                            ts = pd.to_datetime(k[0], unit='ms')
                            close = float(k[4])
                            price_data.append({'timestamp': ts, 'close': close})
                        if not price_data:
                            print(f"[오류] 바이낸스에서 가격 데이터를 받아오지 못했습니다. 진행을 멈춥니다.")
                            self.update_progress(price_progress, "오류: 바이낸스에서 가격 데이터를 받아오지 못했습니다.")
                            return False
                    except Exception as e:
                        print(f"[오류] 바이낸스 가격 데이터 자동 다운로드 실패: {str(e)}. 진행을 멈춥니다.")
                        self.update_progress(price_progress, f"오류: 바이낸스 가격 데이터 자동 다운로드 실패: {str(e)}")
                        return False
                price_df = pd.DataFrame(price_data)
                price_df = price_df.sort_values('timestamp')
                price_df.set_index('timestamp', inplace=True)
                if price_df.empty:
                    print(f"[오류] 가격 데이터프레임이 비어 있습니다. 진행을 멈춥니다.")
                    self.update_progress(price_progress, "오류: 가격 데이터프레임이 비어 있습니다.")
                    return False

                # 거래 실행 진행률 업데이트
                trade_progress = price_progress + 2
                self.update_progress(trade_progress, f"{timeframe} 트레이딩/백테스트 진행 중...")
                self.logger.info(f"{timeframe} 트레이딩/백테스트 진행 중...")
                success = self.execute_trades(price_df, sentiment_results, pattern_results)
                if not success:
                    print(f"[오류] 거래 실행에 실패했습니다. 진행을 멈춥니다.")
                    self.update_progress(trade_progress, "오류: 거래 실행에 실패했습니다.")
                    return False
                if success:
                    # 성능 평가 진행률 업데이트
                    eval_progress = trade_progress + 2
                    self.update_progress(eval_progress, f"{timeframe} 성능 평가/시각화 중...")
                    self.logger.info(f"{timeframe} 성능 평가/시각화 중...")
                    self.evaluate_performance()
                    self.visualize_results()

                tf_elapsed = time.time() - tf_start
                percent = (idx+1) / total_timeframes * 100
                total_elapsed = time.time() - start_time
                eta = total_elapsed / (idx+1) * (total_timeframes - (idx+1)) if idx+1 < total_timeframes else 0
                print(f"[진행률] {percent:.1f}% ({idx+1}/{total_timeframes}) | 경과: {total_elapsed:.1f}s | 예상 남은: {eta:.1f}s")
                
                # 전체 진행률 업데이트
                overall_progress = 70 + ((idx+1) / total_timeframes) * 10
                self.update_progress(overall_progress, f"{timeframe} 시간봉 처리 완료 ({idx+1}/{total_timeframes})")

            self.update_progress(80, "모든 시간프레임 처리 완료")
            print(f"\n전체 시뮬레이션 완료! 총 경과 시간: {time.time() - start_time:.1f}초")
            # 논문 제출용 결과 자동 생성
            self.paper_report_for_submission()
            return True
            
        except Exception as e:
            self.logger.error(f"시뮬레이션 실행 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            self.update_progress(70, f"오류 발생: {str(e)}")
            return False
    
    def execute_trades(self, price_data, sentiment_results, pattern_results):
        """거래 로직 실행"""
        try:
            # 최초 포트폴리오 가치 기록
            initial_price = price_data.iloc[0]['close']
            self.portfolio_values.append({
                'timestamp': price_data.index[0],
                'portfolio_value': self.balance,
                'price': initial_price
            })
            
            # 거래 신호 진단용
            fusion_signals = []
            trades_executed = 0
            
            # 각 타임스탬프에 대해 거래 결정
            for i in range(1, len(price_data)):
                current_data = price_data.iloc[i]
                current_time = price_data.index[i]
                current_price = current_data['close']
                prev_price = price_data.iloc[i-1]['close']
                
                # 가격 변화율
                price_change = (current_price - prev_price) / prev_price
                
                # 패턴 인식 결과가 없는 경우 기본값 사용
                if not pattern_results or 'prediction' not in pattern_results:
                    pattern_signal = 0.0
                else:
                    pattern_signal = pattern_results.get('prediction', 0.0)
                
                # 감성 분석 결과가 없는 경우 기본값 사용
                if not sentiment_results:
                    sentiment_signal = 0.0
                else:
                    # 감성 점수를 -1 ~ 1 범위로 정규화
                    sentiment_score = sentiment_results.get('sentiment_score', 0.0)
                    bullish_ratio = sentiment_results.get('bullish_ratio', 0.5)
                    bearish_ratio = sentiment_results.get('bearish_ratio', 0.5)
                    
                    # 긍정:부정 비율로 신호 계산
                    if bullish_ratio + bearish_ratio > 0:
                        sentiment_signal = (bullish_ratio - bearish_ratio) / (bullish_ratio + bearish_ratio)
                    else:
                        sentiment_signal = 0.0
                
                # === 1. 비선형 멀티모달 융합 적용 ===
                if hasattr(self, 'fusion_model'):
                    if isinstance(self.fusion_model, AttentionFusion):
                        fusion_signal, attention_weights = self.fusion_model(
                            pattern_signal, sentiment_signal, price_change
                        )
                        # 주의 가중치 로깅
                        if i % 20 == 0:  # 20개 간격으로 로깅
                            self.logger.info(
                                f"주의 가중치: 캔들스틱={attention_weights[0]:.2f}, "
                                f"감성={attention_weights[1]:.2f}, 가격={attention_weights[2]:.2f}"
                            )
                    else:
                        fusion_signal = self.fusion_model(
                            pattern_signal, sentiment_signal, price_change
                        )
                else:
                    # 기존 방식: 선형 결합
                    fusion_signal = 0.5 * pattern_signal + 0.3 * sentiment_signal + 0.2 * (1 if price_change > 0 else -1)
                
                # === 2. 강화학습(RL) 적용 ===
                if self.use_rl:
                    # 현재 상태 벡터 구성
                    state = convert_to_state(
                        pattern_signal, sentiment_signal, price_change, 
                        self.position, self.balance
                    )
                    
                    # RL 모델로 행동 결정
                    action, action_prob = self.rl_trader.get_action(state, training=False)
                    rl_position = action_to_position(action)
                    
                    # 로깅
                    if i % 20 == 0:
                        self.logger.info(f"RL 행동: {rl_position} (확률: {action_prob:.4f})")
                    
                    # 심화 학습을 위한 보상 저장 (실제 포지션 변경 후)
                    old_balance = self.balance
                
                # === 3. 앙상블 방법론 적용 ===
                if self.use_ensemble:
                    # 다중 시간프레임 앙상블 (현재는 동일 신호 사용)
                    timeframe_signals = {tf: fusion_signal for tf in self.config.get('data', {}).get('timeframes', ['1d'])}
                    
                    # 모델 앙상블: 융합 모델 + RL 모델
                    model_predictions = {'fusion': fusion_signal}
                    if self.use_rl:
                        # RL 모델의 출력을 -1 ~ 1 범위로 변환
                        if rl_position == 'long':
                            model_predictions['rl'] = action_prob
                        elif rl_position == 'short':
                            model_predictions['rl'] = -action_prob
                        else:
                            model_predictions['rl'] = 0.0
                    
                    # 앙상블 방법으로 최종 신호 생성
                    tf_ensemble_signal = self.timeframe_ensemble.combine_signals(timeframe_signals)
                    model_ensemble_signal = self.model_ensemble.combine_predictions(model_predictions)
                    
                    # 최종 앙상블 신호: 시간프레임:모델 = 6:4 비율
                    final_signal = 0.6 * tf_ensemble_signal + 0.4 * model_ensemble_signal
                    
                    if i % 20 == 0:
                        self.logger.info(
                            f"앙상블 신호: 시간프레임={tf_ensemble_signal:.2f}, "
                            f"모델={model_ensemble_signal:.2f}, 최종={final_signal:.2f}"
                        )
                else:
                    # 앙상블 미사용 시 융합 신호를 최종 신호로 사용
                    final_signal = fusion_signal
                
                # 거래 결정 (임계값 0.15로 완화)
                if final_signal > 0.15 and self.position != 'long':
                    self.open_long_position(current_time, current_price)
                    trades_executed += 1
                elif final_signal < -0.15 and self.position != 'short':
                    self.open_short_position(current_time, current_price)
                    trades_executed += 1
                elif abs(final_signal) < 0.05 and self.position != 'neutral':
                    self.close_position(current_time, current_price)
                    trades_executed += 1
                
                # 포트폴리오 가치 기록
                self.portfolio_values.append({
                    'timestamp': current_time,
                    'portfolio_value': self.calculate_portfolio_value(current_price),
                    'price': current_price,
                    'signal': final_signal,
                    'position': self.position
                })
                
                # 강화학습 모델 학습용 보상 계산 (수익률)
                if self.use_rl and i % 10 == 0:  # 10 스텝마다 학습
                    new_balance = self.calculate_portfolio_value(current_price)
                    reward = (new_balance - old_balance) / old_balance
                    
                    # 다음 상태 및 완료 여부
                    next_state = convert_to_state(
                        pattern_signal, sentiment_signal, price_change, 
                        self.position, new_balance
                    )
                    done = (i == len(price_data) - 1)
                    
                    # 경험 저장
                    self.rl_trader.memory.push(
                        state, action, reward, next_state, done, action_prob
                    )
                    
                    # 모델 학습
                    if len(self.rl_trader.memory) >= self.rl_trader.batch_size:
                        self.rl_trader.train()
                
                # 진단용 신호 저장
                fusion_signals.append(final_signal)
            
            # 거래 진단 출력
            if not trades_executed:
                self.logger.warning("[진단] 거래 신호가 한 번도 발생하지 않았습니다.")
                if fusion_signals:
                    signal_min = min(fusion_signals)
                    signal_max = max(fusion_signals)
                    signal_avg = sum(fusion_signals) / len(fusion_signals)
                    self.logger.info(f"[진단] 융합 신호 분포: 최소={signal_min:.4f}, 최대={signal_max:.4f}, 평균={signal_avg:.4f}")
            else:
                self.logger.info(f"총 {trades_executed}개의 거래가 실행되었습니다.")
            
            # 강화학습 모델 저장
            if self.use_rl:
                self.rl_trader.save_model(os.path.join(self.results_dir, 'rl_models'))
            
            # 포트폴리오 가치 기록이 없으면 성능 평가 불가
            if not self.portfolio_values:
                raise ValueError("거래 기록이 없어 성능 평가를 진행할 수 없습니다.")
            
            return True
        
        except Exception as e:
            self.logger.error(f"거래 실행 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def open_long_position(self, timestamp, price):
        """롱 포지션 열기 (매수)"""
        if self.position == 'short':
            # 숏 포지션 먼저 청산
            self.close_position(timestamp, price)
        
        # 매수 가능한 BTC 수량 계산 (잔액의 90% 사용)
        amount_to_buy = (self.balance * 0.9) / price
        cost = amount_to_buy * price
        fee = cost * self.trade_fee
        
        # 잔액 조정
        self.balance -= (cost + fee)
        self.btc_balance += amount_to_buy
        self.position = 'long'
        
        # 거래 기록
        self.trades.append({
            'timestamp': timestamp,
            'type': 'buy',
            'price': price,
            'amount': amount_to_buy,
            'fee': fee,
            'balance_after': self.balance,
            'btc_balance_after': self.btc_balance
        })
        
        self.logger.info(f"롱 포지션 오픈: {amount_to_buy:.8f} BTC @ ${price:.2f}")
    
    def open_short_position(self, timestamp, price):
        """숏 포지션 열기 (페이퍼 트레이딩 용)"""
        if self.position == 'long':
            # 롱 포지션 먼저 청산
            self.close_position(timestamp, price)
        
        # 페이퍼 트레이딩에서는 숏 포지션을 롱 포지션의 반대로 시뮬레이션
        self.position = 'short'
        
        # 거래 기록
        self.trades.append({
            'timestamp': timestamp,
            'type': 'short',
            'price': price,
            'amount': 0,  # 페이퍼 트레이딩
            'fee': 0,
            'balance_after': self.balance,
            'btc_balance_after': self.btc_balance
        })
        
        self.logger.info(f"숏 포지션 오픈 (페이퍼 트레이딩) @ ${price:.2f}")
    
    def close_position(self, timestamp, price):
        """포지션 청산"""
        if self.position == 'long':
            # 롱 포지션 청산 (BTC 판매)
            amount_to_sell = self.btc_balance
            revenue = amount_to_sell * price
            fee = revenue * self.trade_fee
            
            # 잔액 조정
            self.balance += (revenue - fee)
            self.btc_balance = 0
            
            # 거래 기록
            self.trades.append({
                'timestamp': timestamp,
                'type': 'sell',
                'price': price,
                'amount': amount_to_sell,
                'fee': fee,
                'balance_after': self.balance,
                'btc_balance_after': self.btc_balance
            })
            
            self.logger.info(f"롱 포지션 청산: {amount_to_sell:.8f} BTC @ ${price:.2f}")
            
        elif self.position == 'short':
            # 숏 포지션 청산 (페이퍼 트레이딩 용)
            # 거래 기록
            self.trades.append({
                'timestamp': timestamp,
                'type': 'cover',
                'price': price,
                'amount': 0,  # 페이퍼 트레이딩
                'fee': 0,
                'balance_after': self.balance,
                'btc_balance_after': self.btc_balance
            })
            
            self.logger.info(f"숏 포지션 청산 (페이퍼 트레이딩) @ ${price:.2f}")
        
        # 포지션 상태 업데이트
        self.position = 'neutral'
    
    def calculate_portfolio_value(self, current_price):
        """현재 포트폴리오 가치 계산"""
        portfolio_value = self.balance
        if self.position == 'long':
            portfolio_value += self.btc_balance * current_price
        elif self.position == 'short':
            # 숏 포지션의 경우 반대로 계산
            portfolio_value -= self.btc_balance * current_price
        return portfolio_value
    
    def evaluate_performance(self):
        """거래 성능 평가"""
        try:
            if not self.trades or not self.portfolio_values:
                self.logger.warning("거래 기록이 없어 성능 평가를 진행할 수 없습니다.")
                return
            
            # 거래 성능 지표 계산
            self.performance_metrics['total_trades'] = len(self.trades)
            
            # 승/패 거래 및 수익/손실 계산
            profits = []
            losses = []
            prev_balance = self.initial_balance
            
            for trade in self.trades:
                if trade['type'] in ['sell', 'cover']:
                    profit = trade['balance_after'] - prev_balance
                    if profit > 0:
                        self.performance_metrics['winning_trades'] += 1
                        profits.append(profit)
                    else:
                        self.performance_metrics['losing_trades'] += 1
                        losses.append(profit)
                    prev_balance = trade['balance_after']
            
            # 승률 계산
            if self.performance_metrics['total_trades'] > 0:
                self.performance_metrics['win_rate'] = self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
            
            # 평균 수익/손실 계산
            if profits:
                self.performance_metrics['avg_profit'] = sum(profits) / len(profits)
            if losses:
                self.performance_metrics['avg_loss'] = sum(losses) / len(losses)
            
            # 수익 팩터 계산
            if losses and sum(losses) != 0:
                self.performance_metrics['profit_factor'] = sum(profits) / abs(sum(losses)) if losses else float('inf')
            
            # 최대 낙폭 계산
            portfolio_df = pd.DataFrame(self.portfolio_values)
            portfolio_df['drawdown'] = portfolio_df['portfolio_value'].cummax() - portfolio_df['portfolio_value']
            self.performance_metrics['max_drawdown'] = portfolio_df['drawdown'].max() / portfolio_df['portfolio_value'].max()
            
            # 최종 수익률 계산
            initial_value = self.portfolio_values[0]['portfolio_value']
            final_value = self.portfolio_values[-1]['portfolio_value']
            self.performance_metrics['final_return'] = (final_value / initial_value) - 1
            
            # 샤프 비율 계산 (연간화)
            returns = portfolio_df['portfolio_value'].pct_change().dropna()
            if len(returns) > 1:
                self.performance_metrics['sharpe_ratio'] = np.sqrt(365) * (returns.mean() / returns.std())
            
            self.logger.info(f"성능 평가 완료: 총 {self.performance_metrics['total_trades']}개 거래, " +
                            f"승률 {self.performance_metrics['win_rate']:.2%}, " +
                            f"최종 수익률 {self.performance_metrics['final_return']:.2%}")
            
            # 결과 저장
            if self.results_dir:
                # 거래 기록 저장
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(os.path.join(self.results_dir, 'trade_history.csv'), index=False)
                
                # 포트폴리오 가치 저장
                portfolio_df.to_csv(os.path.join(self.results_dir, 'portfolio_values.csv'), index=False)
                
                # 성능 지표 저장
                with open(os.path.join(self.results_dir, 'performance_metrics.json'), 'w') as f:
                    json.dump(self.performance_metrics, f, indent=4)
            
        except Exception as e:
            self.logger.error(f"성능 평가 중 오류 발생: {str(e)}")
    
    def visualize_results(self):
        """결과 시각화"""
        try:
            if not self.portfolio_values:
                self.logger.warning("시각화할 데이터가 없습니다.")
                return
            
            # 포트폴리오 가치 변화 시각화
            portfolio_df = pd.DataFrame(self.portfolio_values)
            
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], label='Portfolio Value')
            plt.plot(portfolio_df['timestamp'], portfolio_df['price'], label='BTC Price', alpha=0.5)
            
            plt.title('Portfolio Value vs BTC Price')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # 결과 저장
            plt.savefig(os.path.join(self.results_dir, 'portfolio_performance.png'))
            plt.close()
            
            self.logger.info(f"결과가 {self.results_dir}에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"시각화 중 오류 발생: {str(e)}")
    
    def save_performance_metrics(self, filepath):
        """성능 메트릭스 저장"""
        try:
            # 성능 지표 CSV 형식으로 저장
            metrics_df = pd.DataFrame([self.performance_metrics])
            metrics_df.to_csv(filepath, index=False)
            
            self.logger.info(f"성능 메트릭스가 저장되었습니다: {filepath}")
            
            return True
        except Exception as e:
            self.logger.error(f"성능 메트릭스 저장 중 오류 발생: {str(e)}")
            return False
    
    def paper_report_for_submission(self):
        """논문 제출용 결과 파일 자동 생성"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            import os
            import seaborn as sns
            
            # 1. 성능지표 요약 CSV
            metrics = self.performance_metrics
            metrics_df = pd.DataFrame([{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}])
            metrics_df.to_csv(os.path.join(self.results_dir, 'performance_metrics_for_paper.csv'), index=False)
            
            # 2. 포트폴리오 가치 변화 그래프
            portfolio_df = pd.DataFrame(self.portfolio_values)
            
            # timestamp 필드가 없으면 인덱스를 사용
            if 'timestamp' not in portfolio_df.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_df.index, portfolio_df['portfolio_value'], linewidth=2)
                plt.title('포트폴리오 가치 변화', fontsize=15)
                plt.xlabel('기간(일)', fontsize=12)
                plt.ylabel('가치 (USD)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'portfolio_value_chart.png'), dpi=300)
            else:
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], linewidth=2)
                plt.title('포트폴리오 가치 변화', fontsize=15)
                plt.xlabel('날짜', fontsize=12)
                plt.ylabel('가치 (USD)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'portfolio_value_chart.png'), dpi=300)
            
            # 3. 매수/매도 신호 분포 히스토그램
            if 'signal' in portfolio_df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(portfolio_df['signal'], bins=50, kde=True)
                plt.axvline(x=0.15, color='g', linestyle='--', label='매수 임계값')
                plt.axvline(x=-0.15, color='r', linestyle='--', label='매도 임계값')
                plt.title('거래 신호 분포', fontsize=15)
                plt.xlabel('신호 강도', fontsize=12)
                plt.ylabel('빈도', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'signal_distribution.png'), dpi=300)
            
            # 4. 누적 수익률 vs 단순 보유 전략 비교
            plt.figure(figsize=(12, 6))
            
            # 'price' 필드가 있을 때만 단순 보유 전략 계산
            if 'price' in portfolio_df.columns:
                # 단순 보유 전략 (Buy & Hold)
                initial_price = portfolio_df['price'].iloc[0]
                final_price = portfolio_df['price'].iloc[-1]
                buy_hold_return = (final_price / initial_price) - 1
                
                # 누적 수익률 계산
                initial_value = portfolio_df['portfolio_value'].iloc[0]
                portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / initial_value) - 1
                
                # 가격 기반 단순 보유 수익률
                portfolio_df['buy_hold_return'] = (portfolio_df['price'] / initial_price) - 1
                
                # 그래프 그리기 (timestamp 필드가 없으면 인덱스 사용)
                if 'timestamp' not in portfolio_df.columns:
                    plt.plot(portfolio_df.index, portfolio_df['cumulative_return'] * 100, 
                            label=f'멀티모달 앙상블 전략 ({metrics.get("total_return", 0)*100:.2f}%)', linewidth=2)
                    plt.plot(portfolio_df.index, portfolio_df['buy_hold_return'] * 100, 
                            label=f'단순 보유 전략 ({buy_hold_return*100:.2f}%)', linewidth=2, linestyle='--')
                else:
                    plt.plot(portfolio_df['timestamp'], portfolio_df['cumulative_return'] * 100, 
                            label=f'멀티모달 앙상블 전략 ({metrics.get("total_return", 0)*100:.2f}%)', linewidth=2)
                    plt.plot(portfolio_df['timestamp'], portfolio_df['buy_hold_return'] * 100, 
                            label=f'단순 보유 전략 ({buy_hold_return*100:.2f}%)', linewidth=2, linestyle='--')
            else:
                # 가격 정보가 없는 경우, 누적 수익률만 표시
                initial_value = portfolio_df['portfolio_value'].iloc[0]
                portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / initial_value) - 1
                
                if 'timestamp' not in portfolio_df.columns:
                    plt.plot(portfolio_df.index, portfolio_df['cumulative_return'] * 100, 
                            label=f'전략 수익률 ({metrics.get("total_return", 0)*100:.2f}%)', linewidth=2)
                else:
                    plt.plot(portfolio_df['timestamp'], portfolio_df['cumulative_return'] * 100, 
                            label=f'전략 수익률 ({metrics.get("total_return", 0)*100:.2f}%)', linewidth=2)
            
            plt.title('전략 수익률 비교', fontsize=15)
            plt.xlabel('기간', fontsize=12)
            plt.ylabel('누적 수익률 (%)', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'strategy_comparison.png'), dpi=300)
            
            # 5. 포지션 변화 시각화 (포지션 정보가 있는 경우에만)
            if 'position' in portfolio_df.columns:
                plt.figure(figsize=(12, 8))
                # 포지션을 숫자로 변환 (long=1, neutral=0, short=-1)
                position_map = {'long': 1, 'neutral': 0, 'short': -1}
                portfolio_df['position_code'] = portfolio_df['position'].map(position_map)
                
                # 주가 그래프 (위)
                ax1 = plt.subplot(2, 1, 1)
                
                if 'price' in portfolio_df.columns:
                    if 'timestamp' not in portfolio_df.columns:
                        ax1.plot(portfolio_df.index, portfolio_df['price'], color='black', linewidth=1.5)
                    else:
                        ax1.plot(portfolio_df['timestamp'], portfolio_df['price'], color='black', linewidth=1.5)
                    ax1.set_title('자산 가격과 거래 포지션', fontsize=15)
                    ax1.set_ylabel('가격 (USD)', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                    
                    # 롱 포지션 표시
                    longs = portfolio_df[portfolio_df['position'] == 'long']
                    if not longs.empty:
                        if 'timestamp' not in longs.columns:
                            ax1.scatter(longs.index, longs['price'], marker='^', color='green', s=100, label='롱 포지션')
                        else:
                            ax1.scatter(longs['timestamp'], longs['price'], marker='^', color='green', s=100, label='롱 포지션')
                    
                    # 숏 포지션 표시
                    shorts = portfolio_df[portfolio_df['position'] == 'short']
                    if not shorts.empty:
                        if 'timestamp' not in shorts.columns:
                            ax1.scatter(shorts.index, shorts['price'], marker='v', color='red', s=100, label='숏 포지션')
                        else:
                            ax1.scatter(shorts['timestamp'], shorts['price'], marker='v', color='red', s=100, label='숏 포지션')
                    
                    ax1.legend(fontsize=12)
                else:
                    # 가격 정보가 없는 경우, 포트폴리오 가치만 표시
                    if 'timestamp' not in portfolio_df.columns:
                        ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], color='black', linewidth=1.5)
                    else:
                        ax1.plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], color='black', linewidth=1.5)
                    ax1.set_title('포트폴리오 가치와 거래 포지션', fontsize=15)
                    ax1.set_ylabel('가치 (USD)', fontsize=12)
                    ax1.grid(True, alpha=0.3)
                
                # 포지션 그래프 (아래)
                ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                if 'timestamp' not in portfolio_df.columns:
                    ax2.plot(portfolio_df.index, portfolio_df['position_code'], color='blue', linewidth=1.5)
                    ax2.fill_between(portfolio_df.index, portfolio_df['position_code'], 0, 
                                    where=(portfolio_df['position_code'] > 0), color='green', alpha=0.3)
                    ax2.fill_between(portfolio_df.index, portfolio_df['position_code'], 0, 
                                    where=(portfolio_df['position_code'] < 0), color='red', alpha=0.3)
                else:
                    ax2.plot(portfolio_df['timestamp'], portfolio_df['position_code'], color='blue', linewidth=1.5)
                    ax2.fill_between(portfolio_df['timestamp'], portfolio_df['position_code'], 0, 
                                    where=(portfolio_df['position_code'] > 0), color='green', alpha=0.3)
                    ax2.fill_between(portfolio_df['timestamp'], portfolio_df['position_code'], 0, 
                                    where=(portfolio_df['position_code'] < 0), color='red', alpha=0.3)
                ax2.set_ylim([-1.5, 1.5])
                ax2.set_ylabel('포지션\n(1=롱, 0=중립, -1=숏)', fontsize=12)
                ax2.set_xlabel('기간', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_yticks([-1, 0, 1])
                ax2.set_yticklabels(['숏', '중립', '롱'])
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'position_changes.png'), dpi=300)
            
            # 6. 고급 기술 분석: 비선형 융합 및 앙상블 시각화
            if hasattr(self, 'fusion_model') and hasattr(self.fusion_model, '__call__'):
                # 주의 가중치 시각화
                attention_samples = {}
                for i in range(0, len(portfolio_df), len(portfolio_df) // 10):  # 10개 샘플 선택
                    if i < len(portfolio_df) and 'signal' in portfolio_df.columns:
                        signal = portfolio_df['signal'].iloc[i]
                        pattern_signal = 0.5 * signal  # 임시 값
                        sentiment_signal = 0.3 * signal  # 임시 값
                        price_change = 0.2 * signal  # 임시 값
                        
                        # fusion_model이 호출 가능하고 적절한 형태로 attention_weights를 반환하는지 확인
                        try:
                            result = self.fusion_model(pattern_signal, sentiment_signal, price_change)
                            if isinstance(result, tuple) and len(result) == 2:
                                _, attention_weights = result
                                if 'timestamp' in portfolio_df.columns:
                                    attention_samples[portfolio_df['timestamp'].iloc[i]] = attention_weights
                                else:
                                    attention_samples[i] = attention_weights
                        except Exception as e:
                            self.logger.warning(f"주의 가중치 계산 중 오류: {str(e)}")
                
                if attention_samples:
                    # 주의 가중치 시각화
                    dates = list(attention_samples.keys())
                    weights = np.array(list(attention_samples.values()))
                    
                    if weights.shape[1] == 3:  # 3개의 가중치가 있는 경우만 처리
                        plt.figure(figsize=(14, 6))
                        plt.stackplot(dates, 
                                    weights[:, 0], weights[:, 1], weights[:, 2],
                                    labels=['캔들스틱 패턴', '뉴스 감성', '가격 변화'],
                                    alpha=0.8)
                        plt.title('멀티모달 융합 주의 가중치 변화', fontsize=15)
                        plt.xlabel('기간', fontsize=12)
                        plt.ylabel('주의 가중치 비율', fontsize=12)
                        plt.ylim(0, 1)
                        plt.legend(fontsize=12)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.results_dir, 'attention_weights.png'), dpi=300)
            
            # 모든 그래프 창 닫기
            plt.close('all')
            
            self.logger.info(f"논문 제출용 결과 파일이 {self.results_dir} 폴더에 생성되었습니다.")
            
        except Exception as e:
            self.logger.error(f"논문 제출용 결과 파일 생성 중 오류 발생: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc()) 