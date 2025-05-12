import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
import random
from collections import deque
import json

# DQN용 신경망 모델 정의
class DQNNetwork(nn.Module):
    """
    DQN용 신경망 모델
    - 입력: 상태 (이미지 특징 + 뉴스 특징 + 포지션 등)
    - 출력: 각 액션의 Q-값 (매수/매도/홀딩)
    """
    def __init__(self, state_dim=768, action_dim=3, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        # 신경망 아키텍처 설계
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        """
        순전파 (forward propagation)
        
        Args:
            x: 상태 벡터
            
        Returns:
            torch.Tensor: 각 액션의 Q-값
        """
        return self.layers(x)

# Dueling DQN용 신경망 모델
class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN 신경망 모델
    - 장점: 액션 가치(Q)를 상태 가치(V)와 액션 이점(A)으로 분리
    - 상태에서 가치 판단과 액션 선택을 분리하여 더 효율적인 학습
    """
    def __init__(self, state_dim=768, action_dim=3, hidden_dim=256):
        super(DuelingDQNNetwork, self).__init__()
        
        # 공통 특징 추출 레이어
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 상태 가치 스트림 (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 액션 이점 스트림 (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        """
        순전파 (forward propagation)
        
        Args:
            x: 상태 벡터
            
        Returns:
            torch.Tensor: 각 액션의 Q-값 (상태 가치 + 액션 이점)
        """
        features = self.feature_layer(x)
        
        # 상태 가치 (V)
        value = self.value_stream(features)
        
        # 액션 이점 (A)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A)) -> 안정적인 학습을 위한 정규화
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

class DQNAgent:
    """
    DQN (Deep Q-Network) 에이전트
    - 경험 리플레이와 타겟 네트워크를 통한 안정적인 학습
    - 입실론-그리디 탐색 정책으로 탐색과 활용 균형
    """
    def __init__(self, state_dim=768, action_dim=3, gamma=0.99, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.0005, 
                 batch_size=64, memory_size=10000, target_update_freq=10,
                 use_prioritized_replay=True, use_dueling=False,
                 use_double=False, device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        # 기본 파라미터 설정
        self.state_dim = state_dim  # 상태 차원
        self.action_dim = action_dim  # 행동 수 (매수/매도/홀딩)
        self.gamma = gamma  # 할인 계수
        self.epsilon = epsilon  # 탐색 확률
        self.epsilon_min = epsilon_min  # 최소 탐색 확률
        self.epsilon_decay = epsilon_decay  # 탐색 확률 감소율
        self.learning_rate = learning_rate  # 학습률
        self.batch_size = batch_size  # 배치 크기
        self.memory_size = memory_size  # 메모리 크기
        self.target_update_freq = target_update_freq  # 타겟 네트워크 업데이트 주기
        self.device = device  # 학습 장치 (GPU/CPU)
        
        # 에이전트 이름 및 아키텍처 설정
        self.use_dueling = use_dueling  # Dueling DQN 사용 여부
        self.use_double = use_double  # Double DQN 사용 여부
        self.use_prioritized_replay = use_prioritized_replay  # 우선순위 경험 재생 사용 여부
        
        # 에이전트 유형에 따른 이름 설정
        if self.use_dueling and self.use_double:
            self.name = "Dueling Double DQN"
        elif self.use_dueling:
            self.name = "Dueling DQN"
        elif self.use_double:
            self.name = "Double DQN"
        else:
            self.name = "DQN"
        
        # 메모리(경험 리플레이) 초기화
        if self.use_prioritized_replay:
            from paper1.rl_trader import PrioritizedMemory
            self.memory = PrioritizedMemory(capacity=memory_size)
        else:
            from paper1.rl_trader import Memory
            self.memory = Memory(capacity=memory_size)
        
        # 네트워크 선택 (일반 또는 Dueling)
        if self.use_dueling:
            self.policy_net = DuelingDQNNetwork(state_dim, action_dim).to(device)
            self.target_net = DuelingDQNNetwork(state_dim, action_dim).to(device)
        else:
            self.policy_net = DQNNetwork(state_dim, action_dim).to(device)
            self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        
        # 타겟 네트워크 가중치 초기화 (정책 네트워크와 동일하게)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 평가 모드로 설정
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 학습 통계
        self.train_count = 0
        self.step_count = 0
        
        # 로깅 설정
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # 로그 핸들러 추가 (중복 방지)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.logger.info(f"{self.name} 에이전트 초기화: state_dim={state_dim}, action_dim={action_dim}")
        
    def get_action(self, state, training=True):
        """
        현재 상태에서 액션 선택
        
        Args:
            state: 현재 상태
            training: 학습 모드 여부 (True: 입실론-그리디 사용, False: 그리디)
            
        Returns:
            int: 선택된 액션
            float: 액션에 대한 확률 (로깅용)
        """
        # 탐색 vs 활용 결정
        if training and np.random.rand() < self.epsilon:
            # 탐색: 무작위 액션 선택
            action = np.random.randint(self.action_dim)
            action_probs = np.ones(self.action_dim) / self.action_dim  # 균등 확률
            return action, action_probs[action]
        else:
            # 활용: 신경망을 통한 최적 액션 선택
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
                
                # 소프트맥스를 통한 액션 확률 계산 (로깅용)
                probs = F.softmax(q_values, dim=1).squeeze().cpu().numpy()
                
                return action, probs[action]
    
    def update(self, state, action, reward, next_state, done=False):
        """
        경험 저장 및 학습 수행
        
        Args:
            state: 현재 상태
            action: 수행한 액션
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
            
        Returns:
            float or None: 학습 시 손실값 반환, 학습하지 않은 경우 None
        """
        # 경험 저장
        if self.use_prioritized_replay:
            # 우선순위 경험 재생을 위한 초기 TD 오차 계산
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # 현재 Q값 계산
            current_q = self.policy_net(state_tensor)[0, action].item()
            
            # 다음 상태의 최대 Q값 계산
            if done:
                target_q = reward
            else:
                with torch.no_grad():
                    if self.use_double:
                        # Double DQN: 정책 네트워크로 액션 선택, 타겟 네트워크로 값 평가
                        next_action = self.policy_net(next_state_tensor).argmax(dim=1)
                        target_q = reward + self.gamma * self.target_net(next_state_tensor)[0, next_action].item()
                    else:
                        # 일반 DQN: 타겟 네트워크로 최대 Q값 계산
                        target_q = reward + self.gamma * self.target_net(next_state_tensor).max(dim=1)[0].item()
            
            # TD 오차 계산 (우선순위로 사용)
            td_error = abs(target_q - current_q)
            
            # 경험 저장 (TD 오차와 함께)
            self.memory.push(state, action, reward, next_state, done, 1.0, td_error)
        else:
            # 일반 경험 재생 메모리에 저장
            self.memory.push(state, action, reward, next_state, done, 1.0)
        
        self.step_count += 1
        
        # 일정 횟수 경험이 쌓이면 학습 수행
        loss = None
        if len(self.memory) > self.batch_size:
            loss = self.train()
            
            # 탐색 확률 감소
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # 타겟 네트워크 주기적 업데이트
            if self.train_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.logger.info(f"타겟 네트워크 업데이트: #{self.train_count}, epsilon={self.epsilon:.4f}")
        
        return loss
    
    def train(self):
        """
        경험 재생 메모리에서 배치를 샘플링하여 신경망 학습
        
        Returns:
            float: 손실값
        """
        self.train_count += 1
        
        # 배치 샘플링
        if self.use_prioritized_replay:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, _, indices, weights = self.memory.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, _ = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)  # 가중치 없음
        
        # 텐서 변환 및 장치 이동
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)
        
        # Q값 계산
        q_values = self.policy_net(state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # 타겟 Q값 계산
        with torch.no_grad():
            if self.use_double:
                # Double DQN: 정책 네트워크로 액션 선택, 타겟 네트워크로 값 평가
                next_actions = self.policy_net(next_state_batch).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            else:
                # 일반 DQN: 타겟 네트워크로 최대 Q값 계산
                next_q_values = self.target_net(next_state_batch).max(dim=1)[0]
            
            # 타겟 계산 (종료 상태는 오직 보상만 고려)
            target_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # TD 오차 계산
        td_errors = torch.abs(target_values - q_values).detach().cpu().numpy()
        
        # Huber 손실 함수 (outlier에 강인)
        loss = F.smooth_l1_loss(q_values, target_values, reduction='none')
        
        # 가중치 적용한 손실 계산
        weighted_loss = (loss * weights).mean()
        
        # 역전파 및 최적화
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # 그래디언트 클리핑 (폭발 방지)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 우선순위 경험 재생 사용 시 우선순위 업데이트
        if self.use_prioritized_replay:
            self.memory.update_priorities(indices, td_errors)
        
        return weighted_loss.item()
    
    def save_model(self, path='rl_models', name=None):
        """
        모델 저장
        
        Args:
            path: 저장 경로
            name: 모델 이름 (없으면 에이전트 이름 사용)
        """
        if name is None:
            name = self.name.lower().replace(' ', '_')
        
        # 디렉토리 생성
        os.makedirs(path, exist_ok=True)
        
        # 모델 가중치 저장
        model_path = os.path.join(path, f"{name}.pth")
        torch.save(self.policy_net.state_dict(), model_path)
        
        # 에이전트 설정 저장
        config_path = os.path.join(path, f"{name}_config.json")
        config = {
            'name': self.name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'use_dueling': self.use_dueling,
            'use_double': self.use_double,
            'use_prioritized_replay': self.use_prioritized_replay,
            'train_count': self.train_count
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        self.logger.info(f"모델 저장 완료: {model_path}, 설정: {config_path}")
    
    def load_model(self, path='rl_models', name=None):
        """
        모델 로드
        
        Args:
            path: 로드 경로
            name: 모델 이름 (없으면 에이전트 이름 사용)
            
        Returns:
            bool: 로드 성공 여부
        """
        if name is None:
            name = self.name.lower().replace(' ', '_')
        
        # 모델 파일 경로
        model_path = os.path.join(path, f"{name}.pth")
        config_path = os.path.join(path, f"{name}_config.json")
        
        try:
            # 설정 파일 로드
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 기존 설정 업데이트
                self.epsilon = config.get('epsilon', self.epsilon)
                self.train_count = config.get('train_count', 0)
                
                self.logger.info(f"설정 로드 완료: {config_path}")
            
            # 모델 가중치 로드
            state_dict = torch.load(model_path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            
            self.logger.info(f"모델 로드 완료: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            return False

# 테스트 함수
def test_dqn_agent():
    # 환경 시뮬레이션
    state_dim = 768
    action_dim = 3
    
    # DQN 에이전트 초기화
    agent = DQNAgent(state_dim, action_dim)
    
    # 간단한 에피소드 시뮬레이션
    state = np.random.rand(state_dim)
    for step in range(10):
        # 액션 선택
        action, action_prob = agent.get_action(state)
        
        # 환경과 상호작용 시뮬레이션
        next_state = np.random.rand(state_dim)
        reward = np.random.uniform(-1, 1)
        done = step == 9  # 마지막 스텝에서 종료
        
        # 경험 저장 및 학습
        loss = agent.update(state, action, reward, next_state, done)
        
        # 다음 상태로 이동
        state = next_state
        
        print(f"Step {step}: action={action}, reward={reward:.4f}, loss={loss}")
    
    # 모델 저장
    agent.save_model()
    
    # 모델 로드 테스트
    new_agent = DQNAgent(state_dim, action_dim)
    new_agent.load_model()
    
    print("DQN 에이전트 테스트 완료!")

if __name__ == "__main__":
    test_dqn_agent() 