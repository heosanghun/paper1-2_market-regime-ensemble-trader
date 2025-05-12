#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
특징 융합 모듈
- 캔들스틱 이미지 특징과 뉴스 감성 특징을 융합
- 다양한 융합 방식 지원 (연결, 어텐션, 크로스모달)
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('FeatureFusion')

class FeatureFusionModel(nn.Module):
    """특징 융합 모델"""
    
    def __init__(self, candlestick_dim, sentiment_dim, fusion_layers, output_dim=3):
        """초기화"""
        super(FeatureFusionModel, self).__init__()
        
        # 입력 차원
        self.candlestick_dim = candlestick_dim
        self.sentiment_dim = sentiment_dim
        
        # 융합 차원
        self.fusion_dim = candlestick_dim + sentiment_dim
        
        # 융합 레이어
        self.fusion_layers = nn.ModuleList()
        
        # 첫 번째 레이어
        self.fusion_layers.append(nn.Linear(self.fusion_dim, fusion_layers[0]))
        
        # 중간 레이어
        for i in range(1, len(fusion_layers)):
            self.fusion_layers.append(nn.Linear(fusion_layers[i-1], fusion_layers[i]))
        
        # 출력 레이어
        self.output_layer = nn.Linear(fusion_layers[-1], output_dim)
        
        # 활성화 함수
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, candlestick_features, sentiment_features):
        """순전파"""
        # 특징 연결
        x = torch.cat((candlestick_features, sentiment_features), dim=1)
        
        # 융합 레이어
        for layer in self.fusion_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        
        # 출력 레이어
        x = self.output_layer(x)
        
        return x

class AttentionFusionModel(nn.Module):
    """어텐션 기반 특징 융합 모델"""
    
    def __init__(self, candlestick_dim, sentiment_dim, fusion_layers, output_dim=3):
        """초기화"""
        super(AttentionFusionModel, self).__init__()
        
        # 입력 차원
        self.candlestick_dim = candlestick_dim
        self.sentiment_dim = sentiment_dim
        
        # 어텐션 가중치
        self.candlestick_attention = nn.Linear(candlestick_dim, 1)
        self.sentiment_attention = nn.Linear(sentiment_dim, 1)
        
        # 융합 차원
        self.fusion_dim = candlestick_dim + sentiment_dim
        
        # 융합 레이어
        self.fusion_layers = nn.ModuleList()
        
        # 첫 번째 레이어
        self.fusion_layers.append(nn.Linear(self.fusion_dim, fusion_layers[0]))
        
        # 중간 레이어
        for i in range(1, len(fusion_layers)):
            self.fusion_layers.append(nn.Linear(fusion_layers[i-1], fusion_layers[i]))
        
        # 출력 레이어
        self.output_layer = nn.Linear(fusion_layers[-1], output_dim)
        
        # 활성화 함수
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, candlestick_features, sentiment_features):
        """순전파"""
        # 어텐션 가중치 계산
        c_att = torch.sigmoid(self.candlestick_attention(candlestick_features))
        s_att = torch.sigmoid(self.sentiment_attention(sentiment_features))
        
        # 가중치 적용
        c_features = candlestick_features * c_att
        s_features = sentiment_features * s_att
        
        # 특징 연결
        x = torch.cat((c_features, s_features), dim=1)
        
        # 융합 레이어
        for layer in self.fusion_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        
        # 출력 레이어
        x = self.output_layer(x)
        
        return x

class CrossModalFusionModel(nn.Module):
    """크로스모달 특징 융합 모델"""
    
    def __init__(self, candlestick_dim, sentiment_dim, fusion_layers, output_dim=3):
        """초기화"""
        super(CrossModalFusionModel, self).__init__()
        
        # 입력 차원
        self.candlestick_dim = candlestick_dim
        self.sentiment_dim = sentiment_dim
        
        # 크로스모달 어텐션
        self.c_to_s = nn.Linear(candlestick_dim, sentiment_dim)
        self.s_to_c = nn.Linear(sentiment_dim, candlestick_dim)
        
        # 융합 차원
        self.fusion_dim = candlestick_dim + sentiment_dim
        
        # 융합 레이어
        self.fusion_layers = nn.ModuleList()
        
        # 첫 번째 레이어
        self.fusion_layers.append(nn.Linear(self.fusion_dim, fusion_layers[0]))
        
        # 중간 레이어
        for i in range(1, len(fusion_layers)):
            self.fusion_layers.append(nn.Linear(fusion_layers[i-1], fusion_layers[i]))
        
        # 출력 레이어
        self.output_layer = nn.Linear(fusion_layers[-1], output_dim)
        
        # 활성화 함수
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, candlestick_features, sentiment_features):
        """순전파"""
        # 크로스모달 어텐션
        c_to_s_att = torch.sigmoid(self.c_to_s(candlestick_features))
        s_to_c_att = torch.sigmoid(self.s_to_c(sentiment_features))
        
        # 크로스모달 특징
        enhanced_c = candlestick_features + s_to_c_att
        enhanced_s = sentiment_features + c_to_s_att
        
        # 특징 연결
        x = torch.cat((enhanced_c, enhanced_s), dim=1)
        
        # 융합 레이어
        for layer in self.fusion_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        
        # 출력 레이어
        x = self.output_layer(x)
        
        return x

class FeatureFusion:
    """특징 융합 클래스"""
    
    def __init__(self, config):
        """초기화"""
        self.config = config
        
        # 융합 방식
        fusion_config = config.get('fusion_model', {})
        self.fusion_method = fusion_config.get('fusion_method', 'concat')
        
        # 융합 레이어
        self.fusion_layers = fusion_config.get('fusion_layers', [256, 128, 64])
        
        # 출력 차원
        self.output_dim = fusion_config.get('output_dim', 3)
        
        # GPU 사용 여부
        self.use_gpu = config.get('use_gpu', True) and torch.cuda.is_available()
        
        # 모델 초기화
        self.model = None
        self._initialize_model()
        
        logger.info(f"특징 융합 모델 초기화 완료 (방식: {self.fusion_method})")
    
    def _initialize_model(self):
        """모델 초기화"""
        # 캔들스틱 특징 차원
        candlestick_dim = 256  # 기본값
        
        # 감성 특징 차원
        sentiment_dim = 128  # 기본값
        
        # 융합 방식
        if self.fusion_method == 'concat':
            self.model = FeatureFusionModel(
                candlestick_dim, 
                sentiment_dim, 
                self.fusion_layers, 
                self.output_dim
            )
        elif self.fusion_method == 'attention':
            self.model = AttentionFusionModel(
                candlestick_dim, 
                sentiment_dim, 
                self.fusion_layers, 
                self.output_dim
            )
        elif self.fusion_method == 'cross-modal':
            self.model = CrossModalFusionModel(
                candlestick_dim, 
                sentiment_dim, 
                self.fusion_layers, 
                self.output_dim
            )
        else:
            logger.warning(f"알 수 없는 융합 방식: {self.fusion_method}. 기본 연결 방식으로 초기화합니다.")
            self.model = FeatureFusionModel(
                candlestick_dim, 
                sentiment_dim, 
                self.fusion_layers, 
                self.output_dim
            )
        
        # GPU 설정
        if self.use_gpu:
            self.model.cuda()
    
    def fuse(self, candlestick_features, sentiment_features):
        """특징 융합"""
        if isinstance(sentiment_features, dict):
            # 감성 분석 결과 딕셔너리를 특징 벡터로 변환
            logger.info(f"감성 분석 결과를 특징 벡터로 변환합니다")
            
            # 딕셔너리에서 수치형 값 추출하여 특징 벡터로 변환
            sentiment_vector = []
            
            # 핵심 수치 값 추출
            if 'sentiment_score' in sentiment_features:
                sentiment_vector.append(sentiment_features['sentiment_score'])
            if 'bullish_ratio' in sentiment_features:
                sentiment_vector.append(sentiment_features['bullish_ratio'])
            if 'bearish_ratio' in sentiment_features:
                sentiment_vector.append(sentiment_features['bearish_ratio'])
            if 'neutral_ratio' in sentiment_features:
                sentiment_vector.append(sentiment_features['neutral_ratio'])
                
            # 기본 값 추가 (충분한 차원 확보)
            while len(sentiment_vector) < 128:
                sentiment_vector.append(0.0)
                
            # 감성 특징으로 변환
            sentiment_features = [sentiment_vector for _ in range(len(candlestick_features))]
            
        logger.info(f"캔들스틱 특징({len(candlestick_features)})과 감성 특징({len(sentiment_features)}) 융합 중...")
        
        # NumPy 배열로 변환
        c_features = np.array(candlestick_features)
        s_features = np.array(sentiment_features)
        
        # 특징 변환
        c_features = torch.FloatTensor(c_features)
        s_features = torch.FloatTensor(s_features)
        
        # GPU 이동
        if self.use_gpu:
            c_features = c_features.cuda()
            s_features = s_features.cuda()
        
        # 모델 추론 모드
        self.model.eval()
        
        with torch.no_grad():
            # 특징 융합
            fused_features = self.model(c_features, s_features)
            
            # CPU로 이동 (필요시)
            if self.use_gpu:
                fused_features = fused_features.cpu()
            
            # 넘파이 변환
            fused_features = fused_features.numpy()
        
        logger.info(f"특징 융합 완료. 융합 특징 차원: {fused_features.shape}")
        
        return fused_features 