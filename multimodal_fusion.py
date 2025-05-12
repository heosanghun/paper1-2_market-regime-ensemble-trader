import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionFusion(nn.Module):
    """
    주의 메커니즘을 활용한 멀티모달 융합 모듈
    - 캔들스틱 이미지 특징
    - 뉴스 감성 특징
    - 가격 데이터 특징 
    을 융합하여 최종 거래 신호를 생성
    """
    def __init__(self, feature_dim=64):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        
        # 각 모달리티별 특징 추출기
        self.candlestick_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.price_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 주의 가중치 생성기
        self.attention = nn.Sequential(
            nn.Linear(feature_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        # 최종 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Tanh()  # -1 ~ 1 사이의 거래 신호 생성
        )
    
    def forward(self, candlestick_signal, sentiment_signal, price_signal):
        # 각 신호를 텐서로 변환
        if not isinstance(candlestick_signal, torch.Tensor):
            candlestick_signal = torch.tensor([candlestick_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(sentiment_signal, torch.Tensor):
            sentiment_signal = torch.tensor([sentiment_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(price_signal, torch.Tensor):
            price_signal = torch.tensor([price_signal], dtype=torch.float32).unsqueeze(1)
        
        # 각 모달리티별 특징 추출
        candlestick_feature = self.candlestick_encoder(candlestick_signal)
        sentiment_feature = self.sentiment_encoder(sentiment_signal)
        price_feature = self.price_encoder(price_signal)
        
        # 모든 특징 결합
        combined_features = torch.cat([candlestick_feature, sentiment_feature, price_feature], dim=1)
        
        # 주의 가중치 생성
        attention_weights = self.attention(combined_features.view(-1, self.feature_dim * 3))
        
        # 주의 가중치 적용
        weighted_candlestick = candlestick_feature * attention_weights[:, 0].unsqueeze(1)
        weighted_sentiment = sentiment_feature * attention_weights[:, 1].unsqueeze(1)
        weighted_price = price_feature * attention_weights[:, 2].unsqueeze(1)
        
        # 가중 특징 결합
        weighted_features = torch.cat([weighted_candlestick, weighted_sentiment, weighted_price], dim=1)
        
        # 최종 융합 신호 생성
        fusion_signal = self.fusion_layer(weighted_features)
        
        return fusion_signal.item(), attention_weights.detach().cpu().numpy()[0]

class TransformerFusion(nn.Module):
    """
    트랜스포머 기반 멀티모달 융합 모듈
    - 자기주의(self-attention) 메커니즘으로 모달리티 간 관계 학습
    """
    def __init__(self, feature_dim=64, num_heads=4):
        super(TransformerFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 각 모달리티별 특징 추출기
        self.candlestick_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        self.price_encoder = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # 멀티헤드 어텐션
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 피드포워드 네트워크
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # 최종 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Tanh()  # -1 ~ 1 사이의 거래 신호 생성
        )
    
    def forward(self, candlestick_signal, sentiment_signal, price_signal):
        # 각 신호를 텐서로 변환
        if not isinstance(candlestick_signal, torch.Tensor):
            candlestick_signal = torch.tensor([candlestick_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(sentiment_signal, torch.Tensor):
            sentiment_signal = torch.tensor([sentiment_signal], dtype=torch.float32).unsqueeze(1)
        if not isinstance(price_signal, torch.Tensor):
            price_signal = torch.tensor([price_signal], dtype=torch.float32).unsqueeze(1)
        
        # 각 모달리티별 특징 추출
        candlestick_feature = self.candlestick_encoder(candlestick_signal)
        sentiment_feature = self.sentiment_encoder(sentiment_signal)
        price_feature = self.price_encoder(price_signal)
        
        # 시퀀스 생성 (각 모달리티를 시퀀스의 토큰으로 간주)
        # [배치, 시퀀스 길이, 특징 차원]
        sequence = torch.cat([
            candlestick_feature.unsqueeze(1),
            sentiment_feature.unsqueeze(1),
            price_feature.unsqueeze(1)
        ], dim=1)  # [batch, 3, feature_dim]
        
        # 자기주의 메커니즘 적용
        attn_output, _ = self.multihead_attention(sequence, sequence, sequence)
        
        # 잔차 연결
        attn_output = attn_output + sequence
        
        # 피드포워드 네트워크
        ff_output = self.feed_forward(attn_output)
        
        # 잔차 연결
        output = ff_output + attn_output
        
        # 평균 풀링으로 시퀀스 차원 제거
        pooled_output = output.mean(dim=1)  # [batch, feature_dim]
        
        # 최종 융합 신호 생성
        fusion_signal = self.fusion_layer(pooled_output)
        
        return fusion_signal.item()

class CrossModalAttention(nn.Module):
    """
    크로스 모달 어텐션 기반 멀티모달 융합 모듈
    - 이미지(캔들스틱) 특징과 텍스트(뉴스) 특징 간의 상호 참조 기능
    - 각 모달리티가 다른 모달리티의 맥락을 참조하여 풍부한 표현 학습
    - 컨텍스트 게이팅으로 시장 상황에 따른 모달리티 중요도 동적 조절
    """
    def __init__(self, image_dim=512, text_dim=256, output_dim=768, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.image_dim = image_dim  # 이미지 특징 차원 (캔들스틱 CNN 출력)
        self.text_dim = text_dim    # 텍스트 특징 차원 (뉴스 분석 출력)
        self.output_dim = output_dim  # 출력 특징 차원
        
        # 특징 투영 레이어
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # 크로스 모달 어텐션을 위한 쿼리, 키, 밸류 변환
        self.img_to_text_query = nn.Linear(output_dim, output_dim)
        self.text_to_value = nn.Linear(output_dim, output_dim)
        
        self.text_to_img_query = nn.Linear(output_dim, output_dim)
        self.img_to_value = nn.Linear(output_dim, output_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 컨텍스트 게이팅 메커니즘
        self.context_gate = nn.Sequential(
            nn.Linear(output_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # 최종 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim // 4),
            nn.GELU(),
            nn.Linear(output_dim // 4, 1),
            nn.Tanh()  # -1 ~ 1 사이의 거래 신호 생성
        )
    
    def forward(self, image_features, text_features, market_context=None):
        """
        크로스 모달 어텐션 적용
        
        Args:
            image_features: 이미지 특징 [배치, 이미지_차원]
            text_features: 텍스트 특징 [배치, 텍스트_차원]
            market_context: 시장 컨텍스트 정보 (옵션)
            
        Returns:
            융합된 특징 [배치, 출력_차원]
        """
        batch_size = image_features.size(0)
        
        # 입력 검증 및 차원 변환
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        
        # 특징 투영
        image_proj = self.image_projection(image_features)  # [배치, 출력_차원]
        text_proj = self.text_projection(text_features)     # [배치, 출력_차원]
        
        # 1. 이미지 -> 텍스트 크로스 어텐션 (이미지가 텍스트를 참조)
        img_query = self.img_to_text_query(image_proj)  # [배치, 출력_차원]
        text_value = self.text_to_value(text_proj)      # [배치, 출력_차원]
        
        # 어텐션 스코어 계산: (Q * K^T) / sqrt(d_k)
        img_to_text_attn = torch.bmm(
            img_query.unsqueeze(1),                # [배치, 1, 출력_차원]
            text_value.unsqueeze(2)                # [배치, 출력_차원, 1]
        ).squeeze(-1) / (self.output_dim ** 0.5)   # [배치, 1]
        
        img_to_text_attn = F.softmax(img_to_text_attn, dim=1)
        
        # 이미지의 텍스트 참조 특징
        attended_text = text_value * img_to_text_attn.unsqueeze(-1)  # [배치, 출력_차원]
        
        # 2. 텍스트 -> 이미지 크로스 어텐션 (텍스트가 이미지를 참조)
        text_query = self.text_to_img_query(text_proj)  # [배치, 출력_차원]
        img_value = self.img_to_value(image_proj)      # [배치, 출력_차원]
        
        # 어텐션 스코어 계산
        text_to_img_attn = torch.bmm(
            text_query.unsqueeze(1),               # [배치, 1, 출력_차원]
            img_value.unsqueeze(2)                 # [배치, 출력_차원, 1]
        ).squeeze(-1) / (self.output_dim ** 0.5)   # [배치, 1]
        
        text_to_img_attn = F.softmax(text_to_img_attn, dim=1)
        
        # 텍스트의 이미지 참조 특징
        attended_img = img_value * text_to_img_attn.unsqueeze(-1)  # [배치, 출력_차원]
        
        # 참조 특징과 원본 특징 합치기
        enhanced_img = image_proj + attended_text
        enhanced_text = text_proj + attended_img
        
        # 드롭아웃 적용
        enhanced_img = self.dropout(enhanced_img)
        enhanced_text = self.dropout(enhanced_text)
        
        # 3. 컨텍스트 게이팅으로 모달리티 중요도 동적 조절
        concat_features = torch.cat([enhanced_img, enhanced_text], dim=1)
        gate_weights = self.context_gate(concat_features)
        
        # 게이팅된 특징 계산 (시장 상황에 따라 모달리티 중요도 다르게 적용)
        gated_img = enhanced_img * gate_weights[:, 0].unsqueeze(-1)
        gated_text = enhanced_text * gate_weights[:, 1].unsqueeze(-1)
        
        # 최종 융합 특징
        fused_features = gated_img + gated_text
        
        # 거래 신호 생성
        trading_signal = self.fusion_layer(fused_features)
        
        return trading_signal.item(), gate_weights.detach().cpu().numpy()[0]

class AdvancedMultiModalFusion(nn.Module):
    """
    복합적인 멀티모달 융합 모델
    - 캔들스틱 이미지 특징
    - 뉴스 텍스트 특징 
    - 가격/거래량 등 수치 데이터
    를 통합 처리하는 고급 융합 아키텍처
    """
    def __init__(self, 
                 image_dim=512,       # CNN 출력 차원
                 text_dim=256,        # 텍스트 모델 출력 차원
                 numeric_dim=10,      # 수치 특징 차원
                 hidden_dim=128,      # 내부 표현 차원
                 output_dim=768):     # 최종 출력 차원
        super(AdvancedMultiModalFusion, self).__init__()
        
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.numeric_dim = numeric_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. 이미지-텍스트 크로스 모달 어텐션
        self.cross_modal_attn = CrossModalAttention(
            image_dim=image_dim,
            text_dim=text_dim,
            output_dim=hidden_dim
        )
        
        # 2. 수치 데이터 인코더
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. 모달리티 통합 (이미지-텍스트 + 수치)
        self.integration_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # 4. 최종 예측 헤드 (거래 행동)
        self.action_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 3)  # 3 액션: 매수/매도/홀딩
        )
        
        # 5. 거래 비율 예측 헤드 (포지션 크기)
        self.ratio_head = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()  # 0~1 값 (포트폴리오의 몇 %를 투자할지)
        )
    
    def forward(self, image_features, text_features, numeric_features):
        """
        고급 멀티모달 융합
        
        Args:
            image_features: 이미지 특징 [배치, 이미지_차원]
            text_features: 텍스트 특징 [배치, 텍스트_차원]
            numeric_features: 수치 특징 [배치, 수치_차원]
            
        Returns:
            action_probs: 행동 확률 [배치, 3]
            trade_ratio: 거래 비율 [배치, 1]
            융합 특징 [배치, 출력_차원]
        """
        # 입력 검증 및 차원 변환
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        if numeric_features.dim() == 1:
            numeric_features = numeric_features.unsqueeze(0)
        
        # 1. 이미지-텍스트 크로스 모달 어텐션
        img_text_signal, gate_weights = self.cross_modal_attn(image_features, text_features)
        
        # 크로스 모달 어텐션 출력을 배치 텐서로 변환
        img_text_features = torch.full((image_features.size(0), self.hidden_dim), img_text_signal, 
                                        device=image_features.device)
        
        # 2. 수치 데이터 인코딩
        numeric_encoded = self.numeric_encoder(numeric_features)
        
        # 3. 모달리티 통합
        combined_features = torch.cat([img_text_features, numeric_encoded], dim=1)
        integrated_features = self.integration_layer(combined_features)
        
        # 4. 행동 예측
        action_logits = self.action_head(integrated_features)
        action_probs = F.softmax(action_logits, dim=1)
        
        # 5. 거래 비율 예측
        trade_ratio = self.ratio_head(integrated_features)
        
        return action_probs, trade_ratio, integrated_features, gate_weights

class MultiModalFusion:
    """
    멀티모달 데이터 융합 모듈
    - 캔들차트 패턴, 뉴스 감성, 가격 데이터 등을 통합
    - 다양한 융합 방식 지원 (late fusion, early fusion 등)
    - 시장 레짐 적응형 가중치 조정 기능 추가
    - 멀티 타임프레임 데이터 통합 지원
    """
    
    def __init__(self, fusion_method='attention', feature_dims=None, config=None):
        """
        초기화
        
        Args:
            fusion_method: 융합 방식 ('attention', 'concat', 'weighted_sum', 'gating')
            feature_dims: 각 모달리티의 특징 차원 딕셔너리 (예: {'pattern': 512, 'sentiment': 256, 'price': 32})
            config: 추가 설정
        """
        self.fusion_method = fusion_method
        self.feature_dims = feature_dims or {}
        self.config = config or {}
        
        # 멀티 타임프레임 지원 설정
        self.use_multi_timeframe = self.config.get('use_multi_timeframe', False)
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d'])
        
        # 기본 시간프레임
        self.default_timeframe = self.config.get('default_timeframe', '1h')
        
        # 시간프레임별 데이터 및 가중치
        if self.use_multi_timeframe:
            self.timeframe_weights = {}
            for tf in self.timeframes:
                # 장기 시간프레임에 더 높은 가중치 부여
                if tf == '1m': weight = 0.05
                elif tf == '5m': weight = 0.10
                elif tf == '15m': weight = 0.15
                elif tf == '1h': weight = 0.20
                elif tf == '4h': weight = 0.25
                elif tf == '1d': weight = 0.25
                else: weight = 0.15
                self.timeframe_weights[tf] = weight
        
        # 시장 레짐 적응형 가중치 조정
        self.use_adaptive_weights = self.config.get('use_adaptive_weights', False)
        self.current_market_regime = 'normal'  # 'normal', 'trending', 'volatile', 'ranging'
        
        # 각 모달리티 가중치 초기화
        self.modality_weights = {
            'pattern': 0.4,
            'sentiment': 0.3,
            'price': 0.3
        }
        
        # 모델 초기화
        self._init_model()
    
    def _init_model(self):
        """내부 모델 초기화"""
        # 사용할 모달리티 확인
        self.modalities = list(self.feature_dims.keys()) if self.feature_dims else ['pattern', 'sentiment', 'price']
        
        if self.fusion_method == 'attention':
            self._init_attention_model()
        elif self.fusion_method == 'gating':
            self._init_gating_model()
    
    def _init_attention_model(self):
        """어텐션 기반 융합 모델 초기화"""
        # 실제 프로젝트에서는 torch를 사용하여 attention 모델 구현
        import torch.nn as nn
        
        # 어텐션 가중치 계산용 변수 (간소화된 예시)
        if self.feature_dims:
            self.attention_weights = {modality: 1.0 / len(self.modalities) for modality in self.modalities}
    
    def _init_gating_model(self):
        """게이팅 메커니즘 기반 융합 모델 초기화"""
        # 실제 프로젝트에서는 torch를 사용하여 gating 모델 구현
        pass
    
    def update_market_regime(self, regime, confidence=0.7):
        """
        현재 시장 레짐 업데이트 및 가중치 조정
        
        Args:
            regime: 시장 레짐 ('normal', 'trending', 'volatile', 'ranging')
            confidence: 레짐 분류 신뢰도 (0~1)
        """
        if not self.use_adaptive_weights:
            return
        
        self.current_market_regime = regime
        
        # 레짐에 따른 가중치 조정
        if regime == 'trending' and confidence > 0.6:
            # 추세 시장에서는 가격 패턴 중요성 증가
            self.modality_weights['pattern'] = 0.5
            self.modality_weights['price'] = 0.3
            self.modality_weights['sentiment'] = 0.2
            
            # 시간프레임 가중치 조정
            if self.use_multi_timeframe:
                self._adjust_timeframe_weights_for_regime('trending')
        
        elif regime == 'volatile' and confidence > 0.6:
            # 변동성 시장에서는 감성 분석 중요성 증가
            self.modality_weights['sentiment'] = 0.4
            self.modality_weights['pattern'] = 0.3
            self.modality_weights['price'] = 0.3
            
            # 시간프레임 가중치 조정
            if self.use_multi_timeframe:
                self._adjust_timeframe_weights_for_regime('volatile')
        
        elif regime == 'ranging' and confidence > 0.6:
            # 횡보 시장에서는 패턴과 가격 균등하게 중요
            self.modality_weights['pattern'] = 0.4
            self.modality_weights['price'] = 0.4
            self.modality_weights['sentiment'] = 0.2
            
            # 시간프레임 가중치 조정
            if self.use_multi_timeframe:
                self._adjust_timeframe_weights_for_regime('ranging')
        
        else:  # 'normal' 또는 낮은 신뢰도
            # 기본 가중치로 복원
            self.modality_weights['pattern'] = 0.4
            self.modality_weights['sentiment'] = 0.3
            self.modality_weights['price'] = 0.3
            
            # 시간프레임 가중치 리셋
            if self.use_multi_timeframe:
                self._reset_timeframe_weights()
    
    def _adjust_timeframe_weights_for_regime(self, regime):
        """
        시장 레짐에 따라 시간프레임 가중치 조정
        
        Args:
            regime: 시장 레짐
        """
        if regime == 'trending':
            # 추세 시장에서는 중장기 시간프레임 강화
            weights = {
                '1m': 0.03,
                '5m': 0.07,
                '15m': 0.10,
                '1h': 0.20,
                '4h': 0.30,
                '1d': 0.30
            }
        elif regime == 'volatile':
            # 변동성 시장에서는 단기 시간프레임 강화
            weights = {
                '1m': 0.10,
                '5m': 0.15,
                '15m': 0.20,
                '1h': 0.25,
                '4h': 0.20,
                '1d': 0.10
            }
        elif regime == 'ranging':
            # 횡보 시장에서는 중기 시간프레임 강화
            weights = {
                '1m': 0.05,
                '5m': 0.10,
                '15m': 0.25,
                '1h': 0.30,
                '4h': 0.20,
                '1d': 0.10
            }
        else:
            # 기본 가중치 사용
            return
        
        # 가중치 업데이트
        for tf in self.timeframes:
            if tf in weights:
                self.timeframe_weights[tf] = weights[tf]
    
    def _reset_timeframe_weights(self):
        """시간프레임 가중치를 기본값으로 재설정"""
        default_weights = {
            '1m': 0.05,
            '5m': 0.10,
            '15m': 0.15,
            '1h': 0.20,
            '4h': 0.25,
            '1d': 0.25
        }
        
        for tf in self.timeframes:
            if tf in default_weights:
                self.timeframe_weights[tf] = default_weights[tf]
    
    def fuse_features(self, features_dict, timeframe=None):
        """
        여러 모달리티의 특징을 융합
        
        Args:
            features_dict: 모달리티별 특징 벡터 딕셔너리 (예: {'pattern': [...], 'sentiment': [...], 'price': [...]})
            timeframe: 시간프레임 (멀티 타임프레임 사용 시)
            
        Returns:
            numpy.ndarray: 융합된 특징 벡터
        """
        # 특징 벡터 확인
        if not features_dict:
            raise ValueError("빈 특징 딕셔너리가 전달되었습니다.")
        
        # 멀티 타임프레임 사용 시 처리
        if self.use_multi_timeframe and timeframe:
            if timeframe not in self.timeframes:
                # 지원하지 않는 시간프레임은 기본 시간프레임 가중치 사용
                tf_weight = 1.0
            else:
                tf_weight = self.timeframe_weights.get(timeframe, 1.0)
        else:
            tf_weight = 1.0
        
        # 융합 방식에 따라 처리
        if self.fusion_method == 'concat':
            return self._concat_fusion(features_dict)
        
        elif self.fusion_method == 'weighted_sum':
            return self._weighted_sum_fusion(features_dict, tf_weight)
        
        elif self.fusion_method == 'attention':
            return self._attention_fusion(features_dict, tf_weight)
        
        elif self.fusion_method == 'gating':
            return self._gating_fusion(features_dict, tf_weight)
        
        else:
            # 기본: 단순 연결
            return self._concat_fusion(features_dict)
    
    def fuse_multi_timeframe(self, timeframe_features_dict):
        """
        다중 시간프레임의 특징 벡터 통합
        
        Args:
            timeframe_features_dict: 시간프레임별 특징 딕셔너리
                {timeframe: {modality: features, ...}, ...}
                
        Returns:
            numpy.ndarray: 다중 시간프레임 통합 특징 벡터
        """
        if not self.use_multi_timeframe:
            # 멀티 타임프레임을 사용하지 않는 경우, 기본 시간프레임 사용
            if self.default_timeframe in timeframe_features_dict:
                return self.fuse_features(timeframe_features_dict[self.default_timeframe])
            else:
                # 기본 시간프레임이 없으면 첫 번째 시간프레임 사용
                first_tf = next(iter(timeframe_features_dict))
                return self.fuse_features(timeframe_features_dict[first_tf])
        
        # 시간프레임별 융합 특징 계산
        import numpy as np
        
        # 결과 저장용 리스트
        fused_features_list = []
        weights_list = []
        
        # 각 시간프레임별 융합 수행
        for tf, features_dict in timeframe_features_dict.items():
            # 해당 시간프레임의 가중치 확인
            if tf in self.timeframes:
                tf_weight = self.timeframe_weights[tf]
            else:
                tf_weight = 0.1  # 기본 가중치
            
            # 특징 융합
            fused = self.fuse_features(features_dict, tf)
            
            # 결과 리스트에 추가
            fused_features_list.append(fused)
            weights_list.append(tf_weight)
        
        # 가중 평균 계산
        if not fused_features_list:
            raise ValueError("융합할 특징이 없습니다.")
        
        # 모든 특징 벡터의 차원이 같은지 확인
        feature_dim = fused_features_list[0].shape[0]
        for features in fused_features_list[1:]:
            if features.shape[0] != feature_dim:
                raise ValueError("시간프레임별 특징 벡터의 차원이 일치하지 않습니다.")
        
        # 가중치 정규화
        total_weight = sum(weights_list)
        norm_weights = [w / total_weight for w in weights_list]
        
        # 가중 평균 계산
        result = np.zeros_like(fused_features_list[0])
        for i, features in enumerate(fused_features_list):
            result += features * norm_weights[i]
        
        return result
    
    def _concat_fusion(self, features_dict):
        """특징 벡터 연결 융합"""
        import numpy as np
        
        # 모든 특징 벡터를 연결
        feature_list = []
        for modality in self.modalities:
            if modality in features_dict:
                feature_list.append(features_dict[modality])
        
        # numpy 배열로 변환 후 연결
        return np.concatenate(feature_list)
    
    def _weighted_sum_fusion(self, features_dict, tf_weight=1.0):
        """가중치 합 융합"""
        import numpy as np
        
        # 첫 번째 특징 벡터의 차원 확인
        first_modality = next(iter(features_dict))
        feature_dim = len(features_dict[first_modality])
        
        # 결과 벡터 초기화
        result = np.zeros(feature_dim)
        total_weight = 0
        
        # 각 모달리티의 가중 합 계산
        for modality in self.modalities:
            if modality in features_dict:
                weight = self.modality_weights.get(modality, 1.0 / len(self.modalities))
                # 시간프레임 가중치 적용
                adjusted_weight = weight * tf_weight
                result += features_dict[modality] * adjusted_weight
                total_weight += adjusted_weight
        
        # 정규화
        if total_weight > 0:
            result /= total_weight
        
        return result
    
    def _attention_fusion(self, features_dict, tf_weight=1.0):
        """어텐션 기반 융합"""
        import numpy as np
        
        # 어텐션 가중치 계산 (간소화된 예시)
        attention_weights = {}
        
        # 시장 레짐에 따른 어텐션 조정
        if self.use_adaptive_weights:
            if self.current_market_regime == 'trending':
                # 추세 시장에서는 가격 패턴에 주목
                base_weights = {'pattern': 0.5, 'sentiment': 0.2, 'price': 0.3}
            elif self.current_market_regime == 'volatile':
                # 변동성 시장에서는 감성에 주목
                base_weights = {'pattern': 0.3, 'sentiment': 0.4, 'price': 0.3}
            elif self.current_market_regime == 'ranging':
                # 횡보 시장에서는 골고루 분산
                base_weights = {'pattern': 0.33, 'sentiment': 0.33, 'price': 0.34}
            else:
                # 일반 시장
                base_weights = {'pattern': 0.4, 'sentiment': 0.3, 'price': 0.3}
                
            # 기본 가중치 사용
            for modality in self.modalities:
                if modality in features_dict:
                    attention_weights[modality] = base_weights.get(modality, 0.33)
        else:
            # 동적 어텐션 가중치 계산 (실제 구현은 더 복잡함)
            total_features = 0
            for modality in self.modalities:
                if modality in features_dict:
                    # 특징 벡터의 노름에 기반한 가중치 (간소화)
                    attention_weights[modality] = np.sum(np.abs(features_dict[modality]))
                    total_features += attention_weights[modality]
            
            # 정규화
            if total_features > 0:
                for modality in attention_weights:
                    attention_weights[modality] /= total_features
        
        # 가중 합 계산
        result = np.zeros_like(next(iter(features_dict.values())))
        total_weight = 0
        
        for modality, features in features_dict.items():
            if modality in attention_weights:
                # 어텐션 가중치와 시간프레임 가중치 결합
                weight = attention_weights[modality] * tf_weight
                result += features * weight
                total_weight += weight
        
        # 정규화
        if total_weight > 0:
            result /= total_weight
        
        return result
    
    def _gating_fusion(self, features_dict, tf_weight=1.0):
        """게이팅 메커니즘 기반 융합 (간소화된 구현)"""
        # 실제 구현에서는 학습된 게이팅 네트워크 사용
        # 여기서는 간단한 가중합으로 대체
        return self._weighted_sum_fusion(features_dict, tf_weight)

# 모델 사용 예시
def test_fusion_models():
    # 예시 신호
    candlestick_signal = 0.7  # 캔들스틱 패턴 신호 (예: ResNet50 출력)
    sentiment_signal = 0.3    # 뉴스 감성 신호
    price_signal = 0.1        # 가격 변화율
    
    # 주의 메커니즘 모델
    attention_model = AttentionFusion()
    attention_signal, attention_weights = attention_model(
        candlestick_signal, sentiment_signal, price_signal
    )
    
    print(f"주의 메커니즘 융합 신호: {attention_signal:.4f}")
    print(f"주의 가중치: 캔들스틱={attention_weights[0]:.2f}, 감성={attention_weights[1]:.2f}, 가격={attention_weights[2]:.2f}")
    
    # 트랜스포머 모델
    transformer_model = TransformerFusion()
    transformer_signal = transformer_model(
        candlestick_signal, sentiment_signal, price_signal
    )
    
    print(f"트랜스포머 융합 신호: {transformer_signal:.4f}")
    
    # 크로스 모달 어텐션 모델 테스트
    image_features = torch.rand(1, 512)  # 1개 배치, 512 차원 이미지 특징
    text_features = torch.rand(1, 256)   # 1개 배치, 256 차원 텍스트 특징
    
    cross_modal_model = CrossModalAttention()
    cross_modal_signal, gate_weights = cross_modal_model(image_features, text_features)
    
    print(f"크로스 모달 어텐션 융합 신호: {cross_modal_signal:.4f}")
    print(f"모달리티 게이트 가중치: 이미지={gate_weights[0]:.2f}, 텍스트={gate_weights[1]:.2f}")
    
    # 고급 멀티모달 융합 모델 테스트
    numeric_features = torch.rand(1, 10)  # 1개 배치, 10 차원 수치 특징
    
    advanced_model = AdvancedMultiModalFusion()
    action_probs, trade_ratio, _, gate_weights = advanced_model(
        image_features, text_features, numeric_features
    )
    
    print("\n고급 멀티모달 융합 결과:")
    print(f"행동 확률: 매수={action_probs[0, 0]:.2f}, 홀딩={action_probs[0, 1]:.2f}, 매도={action_probs[0, 2]:.2f}")
    print(f"거래 비율: {trade_ratio.item():.2f}")
    print(f"크로스 모달 게이트 가중치: 이미지={gate_weights[0]:.2f}, 텍스트={gate_weights[1]:.2f}")

if __name__ == "__main__":
    test_fusion_models() 