import torch
import torch.nn as nn
import numpy as np
import os
import logging
import json
from collections import defaultdict
import matplotlib.pyplot as plt

from paper1.dqn_agent import DQNAgent
from paper1.rl_trader import PPOTrader

class AgentEnsemble:
    """
    다양한 강화학습 에이전트의 앙상블을 관리하는 클래스
    - DQN, Double DQN, Dueling DQN, PPO 에이전트 통합
    - 샤프 비율 기반 에이전트 선택 및 가중 투표 메커니즘
    - 하이브리드 접근법: 기술적 지표, 통계적 모델, 심층 강화학습 통합
    - 동적 가중치 조정 및 멀티 타임프레임 분석
    """
    def __init__(self, state_dim=768, action_dim=3, config=None):
        """
        앙상블 에이전트 관리자 초기화
        
        Args:
            state_dim: 상태 차원
            action_dim: 행동 차원 (기본: 매수/홀딩/매도 3가지)
            config: 설정 딕셔너리
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # 로거 설정
        self.logger = logging.getLogger('AgentEnsemble')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        # 모델 저장 경로
        self.models_dir = self.config.get('models_dir', 'models/ensemble')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 학습 장치 설정
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 에이전트 설정
        self.use_dqn = self.config.get('use_dqn', True)
        self.use_double_dqn = self.config.get('use_double_dqn', True)
        self.use_dueling_dqn = self.config.get('use_dueling_dqn', True)
        self.use_ppo = self.config.get('use_ppo', True)
        
        # 앙상블 설정
        self.ensemble_method = self.config.get('ensemble_method', 'weighted_vote')  # weighted_vote, best_agent
        self.performance_metric = self.config.get('performance_metric', 'sharpe')   # sharpe, sortino, returns
        self.update_interval = self.config.get('update_interval', 100)  # 성능 메트릭 업데이트 간격
        self.window_size = self.config.get('window_size', 500)  # 성능 평가 윈도우 크기
        
        # paper1에서는 하이브리드 시스템 사용하지 않음
        self.use_technical_indicators = False
        self.use_statistical_models = False
        self.use_adaptive_weighting = False
        self.use_multi_timeframe = False
        self.risk_management_level = 'none'
        
        # 에이전트 초기화
        self.agents = {}
        self.initialize_agents()
        
        # 성능 지표 추적
        self.performance = defaultdict(list)  # 에이전트별 성능 기록
        self.weights = {}  # 각 에이전트의 가중치
        self.update_counter = 0  # 업데이트 카운터
        
        # 거래 이력 추적
        self.trade_history = []  # 거래 이력 (에이전트별)
        self.portfolio_values = defaultdict(list)  # 에이전트별 포트폴리오 가치
        
        # 시장 레짐
        self.market_regime = "normal"  # 항상 normal로 고정
        
        # 초기화
        self.initialize_weights()
        
        self.logger.info(f"앙상블 에이전트 관리자 초기화 완료: {len(self.agents)}개 에이전트")
    
    def initialize_agents(self):
        """다양한 강화학습 에이전트 초기화"""
        # 1. 기본 DQN 에이전트
        if self.use_dqn:
            self.agents['dqn'] = DQNAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                use_double=False,
                use_dueling=False,
                device=self.device
            )
        
        # 2. Double DQN 에이전트
        if self.use_double_dqn:
            self.agents['double_dqn'] = DQNAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                use_double=True,
                use_dueling=False,
                device=self.device
            )
        
        # 3. Dueling DQN 에이전트
        if self.use_dueling_dqn:
            self.agents['dueling_dqn'] = DQNAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                use_double=True,  # Double DQN + Dueling DQN 조합
                use_dueling=True,
                device=self.device
            )
        
        # 4. PPO 에이전트
        if self.use_ppo:
            self.agents['ppo'] = PPOTrader(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                device=self.device
            )
    
    def initialize_weights(self):
        """
        각 에이전트의 초기 가중치 설정
        기본: 모든 에이전트에 동일한 가중치 부여
        """
        n_agents = len(self.agents)
        if n_agents == 0:
            return
        
        equal_weight = 1.0 / n_agents
        
        for agent_name in self.agents.keys():
            self.weights[agent_name] = equal_weight
            # 성능 기록 초기화
            self.performance[agent_name] = [0.0] * 10  # 초기 성능 지표
    
    def update_weights_based_on_performance(self):
        """
        성능 지표(샤프 비율 등)에 기반하여 에이전트 가중치 업데이트
        """
        if not self.agents:
            return
        
        # 각 에이전트의 성능 지표 계산
        agent_metrics = {}
        
        for agent_name, agent_data in self.portfolio_values.items():
            if len(agent_data) > self.window_size:
                # 수익률 계산 (최근 window_size 기간)
                recent_values = agent_data[-self.window_size:]
                returns = np.diff(recent_values) / recent_values[:-1]
                
                # 성능 지표 계산
                if self.performance_metric == 'sharpe':
                    # 샤프 비율: 수익률의 평균 / 표준편차
                    if np.std(returns) == 0:
                        metric = 0.0
                    else:
                        metric = np.mean(returns) / np.std(returns) * np.sqrt(252)  # 연간화
                
                elif self.performance_metric == 'sortino':
                    # 소르티노 비율: 수익률의 평균 / 하방 표준편차
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
                        metric = 0.0
                    else:
                        metric = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
                
                else:  # 'returns'
                    # 단순 평균 수익률
                    metric = np.mean(returns) * 100  # 퍼센트로 변환
                
                agent_metrics[agent_name] = max(0.0, metric)  # 음수 메트릭은 0으로 처리
            else:
                # 충분한 데이터가 없는 경우 기본값 사용
                agent_metrics[agent_name] = 0.1  # 최소 가중치 보장
        
        # 메트릭이 모두 0인 경우 처리
        if sum(agent_metrics.values()) == 0:
            equal_weight = 1.0 / len(self.agents)
            for agent_name in self.agents.keys():
                self.weights[agent_name] = equal_weight
            return
        
        # 메트릭 기반 가중치 계산
        total_metric = sum(agent_metrics.values())
        
        for agent_name, metric in agent_metrics.items():
            # 메트릭이 높을수록 가중치 증가
            self.weights[agent_name] = metric / total_metric
        
        # 로깅
        self.logger.info(f"에이전트 가중치 업데이트: {[(k, round(v, 3)) for k, v in self.weights.items()]}")
    
    def initialize_hybrid_components(self):
        """하이브리드 시스템 구성요소 초기화 (paper1에서는 사용하지 않음)"""
        # paper1에서는 하이브리드 시스템을 사용하지 않으므로 아무 동작도 하지 않음
        pass
    
    def update_market_regime(self, price_history):
        """
        시장 상황을 분석하여 현재 시장 체제 업데이트 (paper1에서는 사용하지 않음)
        
        Args:
            price_history: 가격 이력 배열
        """
        # paper1에서는 시장 레짐을 사용하지 않으므로 항상 normal로 고정
        self.market_regime = "normal"
    
    def calculate_technical_indicators(self, price_data):
        """
        기술적 지표 계산 (paper1에서는 사용하지 않음)
        
        Args:
            price_data: 가격 데이터 배열
            
        Returns:
            dict: 기본 기술적 지표 값
        """
        return {'signal': 0, 'strength': 0}
    
    def get_action(self, state, price_history=None):
        """
        앙상블 방식으로 최종 행동 결정 (기본 버전)
        
        Args:
            state: 현재 상태
            price_history: 가격 이력 데이터 (사용하지 않음)
            
        Returns:
            int: 선택된 행동
            dict: 에이전트별 행동 및 신뢰도
        """
        if not self.agents:
            # 에이전트가 없으면 기본 행동(홀딩) 반환
            return 1, {'default': (1, 1.0)}
        
        # 각 에이전트의 행동 수집
        agent_actions = {}
        action_votes = [0] * self.action_dim
        
        for agent_name, agent in self.agents.items():
            action, confidence = agent.get_action(state, training=False)
            agent_actions[agent_name] = (action, confidence)
            
            # 가중 투표
            agent_weight = self.weights[agent_name]
            action_votes[action] += agent_weight * confidence
        
        # 앙상블 방식에 따라 최종 행동 결정
        if self.ensemble_method == 'weighted_vote':
            # 가중 투표: 가장 높은 점수를 받은 행동 선택
            final_action = np.argmax(action_votes)
        
        elif self.ensemble_method == 'best_agent':
            # 최고 성능 에이전트 선택
            best_agent = max(self.weights.items(), key=lambda x: x[1])[0]
            final_action = agent_actions[best_agent][0]
        
        else:  # 기본: 다수결
            final_action = np.argmax(action_votes)
        
        # 업데이트 카운터 증가
        self.update_counter += 1
        
        # 정기적인 가중치 업데이트
        if self.update_counter >= self.update_interval:
            self.update_weights_based_on_performance()
            self.update_counter = 0
        
        return final_action, agent_actions
    
    def apply_risk_management(self, action, market_regime, technical_signal):
        """
        리스크 관리 로직 적용 (paper1에서는 사용하지 않음)
        
        Args:
            action: 원래 선택된 행동
            market_regime: 현재 시장 체제
            technical_signal: 기술적 지표 신호
            
        Returns:
            int: 원래 선택된 행동 (변경 없음)
        """
        # paper1에서는 동적 리스크 관리를 사용하지 않음
        return action
    
    def update(self, state, action, reward, next_state, done=False):
        """
        모든 에이전트 업데이트
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        # 보상 스케일링 (하이브리드 개선)
        # 시장 체제에 따라 보상 스케일 조정
        if self.market_regime == "volatile":
            # 변동성 높은 시장에서는 큰 손실 방지 위해 부정적 보상 강화
            if reward < 0:
                reward *= 1.2
        elif self.market_regime == "trending":
            # 추세가 강한 시장에서는 추세 따라가는 보상 강화
            if reward > 0:
                reward *= 1.1
        
        # 각 에이전트 업데이트
        for agent_name, agent in self.agents.items():
            agent.update(state, action, reward, next_state, done)
    
    def record_portfolio_value(self, portfolio_value, market_regime=None):
        """
        현재 포트폴리오 가치 기록
        앙상블 가중치 계산을 위한 데이터 수집
        
        Args:
            portfolio_value: 현재 포트폴리오 가치
            market_regime: 현재 시장 체제 (옵션)
        """
        # 시장 체제 업데이트
        if market_regime:
            self.market_regime = market_regime
        
        # 모든 에이전트에 동일한 포트폴리오 가치 기록
        for agent_name in self.agents.keys():
            self.portfolio_values[agent_name].append(portfolio_value)
    
    def record_trade(self, action, price, amount, timestamp, agent_decisions, technical_signal=None):
        """
        거래 기록
        
        Args:
            action: 수행한 행동
            price: 거래 가격
            amount: 거래 수량
            timestamp: 거래 시간
            agent_decisions: 각 에이전트의 결정
            technical_signal: 기술적 지표 신호 (옵션)
        """
        trade_info = {
            'action': action,
            'price': price,
            'amount': amount,
            'timestamp': timestamp,
            'agent_decisions': agent_decisions,
            'market_regime': self.market_regime
        }
        
        # 기술적 지표 신호 기록 (있는 경우)
        if technical_signal:
            trade_info['technical_signal'] = technical_signal
        
        self.trade_history.append(trade_info)
    
    def save_models(self):
        """모든 에이전트 모델 저장"""
        for agent_name, agent in self.agents.items():
            agent_dir = os.path.join(self.models_dir, agent_name)
            os.makedirs(agent_dir, exist_ok=True)
            agent.save_model(path=agent_dir)
        
        # 앙상블 설정 저장
        ensemble_config = {
            'weights': self.weights,
            'ensemble_method': self.ensemble_method,
            'performance_metric': self.performance_metric,
            'market_regime': self.market_regime,
            'technical_indicators': self.technical_indicators,
            'timeframe_weights': self.timeframe_weights
        }
        
        with open(os.path.join(self.models_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(ensemble_config, f, indent=4)
        
        self.logger.info(f"모든 앙상블 에이전트 모델 저장 완료: {self.models_dir}")
    
    def load_models(self):
        """모든 에이전트 모델 로드"""
        success = True
        
        for agent_name, agent in self.agents.items():
            agent_dir = os.path.join(self.models_dir, agent_name)
            if os.path.exists(agent_dir):
                success &= agent.load_model(path=agent_dir)
        
        # 앙상블 설정 로드
        ensemble_config_path = os.path.join(self.models_dir, 'ensemble_config.json')
        if os.path.exists(ensemble_config_path):
            with open(ensemble_config_path, 'r') as f:
                ensemble_config = json.load(f)
                
                # 가중치 로드 (저장된 에이전트에 맞게 필터링)
                self.weights = {k: v for k, v in ensemble_config.get('weights', {}).items() 
                               if k in self.agents}
                
                # 부족한 가중치는 초기화
                for agent_name in self.agents.keys():
                    if agent_name not in self.weights:
                        self.weights[agent_name] = 1.0 / len(self.agents)
                
                # 방법론 로드
                self.ensemble_method = ensemble_config.get('ensemble_method', self.ensemble_method)
                self.performance_metric = ensemble_config.get('performance_metric', self.performance_metric)
                
                # 하이브리드 시스템 설정 로드
                self.market_regime = ensemble_config.get('market_regime', 'normal')
                if 'technical_indicators' in ensemble_config:
                    self.technical_indicators = ensemble_config['technical_indicators']
                if 'timeframe_weights' in ensemble_config:
                    self.timeframe_weights = ensemble_config['timeframe_weights']
        
        if success:
            self.logger.info(f"모든 앙상블 에이전트 모델 로드 완료: {self.models_dir}")
        else:
            self.logger.warning("일부 앙상블 에이전트 모델 로드 실패")
        
        return success
    
    def visualize_agent_performance(self, save_path=None):
        """
        각 에이전트의 성능 시각화
        
        Args:
            save_path: 이미지 저장 경로 (None이면 저장하지 않음)
        """
        if not self.portfolio_values:
            self.logger.warning("성능 시각화를 위한 데이터가 없습니다.")
            return
        
        plt.figure(figsize=(12, 6))
        
        for agent_name, values in self.portfolio_values.items():
            plt.plot(values, label=f"{agent_name} (weight={self.weights.get(agent_name, 0):.2f})")
        
        plt.title('Agent Performance Comparison')
        plt.xlabel('Trading Steps')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"성능 시각화 이미지 저장: {save_path}")
        
        plt.close()
    
    def get_agent_stats(self):
        """
        각 에이전트의 통계 정보 반환
        
        Returns:
            dict: 에이전트별 통계 정보
        """
        stats = {}
        
        for agent_name, agent in self.agents.items():
            # 기본 정보
            agent_stats = {
                'weight': self.weights.get(agent_name, 0),
                'type': agent_name,
            }
            
            # 성능 지표 (충분한 데이터가 있는 경우)
            if agent_name in self.portfolio_values and len(self.portfolio_values[agent_name]) > 10:
                values = self.portfolio_values[agent_name]
                returns = np.diff(values) / values[:-1]
                
                agent_stats.update({
                    'total_return': (values[-1] / values[0] - 1) * 100 if values[0] > 0 else 0,
                    'mean_return': np.mean(returns) * 100,
                    'volatility': np.std(returns) * 100,
                    'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
                })
            
            stats[agent_name] = agent_stats
        
        return stats

# 테스트 함수
def test_agent_ensemble():
    """에이전트 앙상블 기능 테스트"""
    # 설정
    config = {
        'models_dir': 'models/test_ensemble',
        'ensemble_method': 'weighted_vote',
        'performance_metric': 'sharpe',
        'update_interval': 10,
        'window_size': 50
    }
    
    # 앙상블 초기화
    ensemble = AgentEnsemble(state_dim=768, action_dim=3, config=config)
    
    # 간단한 시뮬레이션
    state = np.random.rand(768)
    portfolio_value = 10000
    
    for step in range(100):
        # 행동 선택
        action, agent_decisions = ensemble.get_action(state)
        
        # 환경과 상호작용 시뮬레이션
        next_state = np.random.rand(768)
        reward = np.random.uniform(-1, 1)
        done = step == 99  # 마지막 스텝에서 종료
        
        # 에이전트 업데이트
        ensemble.update(state, action, reward, next_state, done)
        
        # 포트폴리오 가치 업데이트 시뮬레이션
        portfolio_change = np.random.normal(0.001, 0.01)  # 평균 0.1%, 표준편차 1%
        portfolio_value *= (1 + portfolio_change)
        
        # 결과 기록
        ensemble.record_portfolio_value(portfolio_value)
        ensemble.record_trade(
            action=action,
            price=100 + np.random.normal(0, 1),
            amount=1.0,
            timestamp=step,
            agent_decisions=agent_decisions
        )
        
        # 다음 상태로 이동
        state = next_state
        
        print(f"Step {step}: action={action}, portfolio={portfolio_value:.2f}")
    
    # 각 에이전트 성능 확인
    agent_stats = ensemble.get_agent_stats()
    for agent_name, stats in agent_stats.items():
        print(f"{agent_name}: {stats}")
    
    # 모델 저장 및 로드 테스트
    ensemble.save_models()
    ensemble.load_models()
    
    # 성능 시각화
    ensemble.visualize_agent_performance(save_path='test_ensemble_performance.png')
    
    print("에이전트 앙상블 테스트 완료!")

if __name__ == "__main__":
    test_agent_ensemble() 