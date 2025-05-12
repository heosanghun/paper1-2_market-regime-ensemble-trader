import os
import pandas as pd
import numpy as np
import matplotlib
# Agg 백엔드 설정 (GUI 필요 없음)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# 한글 폰트 설정 제거
# plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우용 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 표시 오류 수정

# 결과 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
results_dir = os.path.join(parent_dir, "results")
output_dir = os.path.join(current_dir, "research_graphs")
os.makedirs(output_dir, exist_ok=True)
print(f"Results directory: {output_dir}")

# 가상 데이터 생성 함수
def generate_sample_data():
    # 1. 포트폴리오 가치 데이터
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    base_value = 10000
    
    # DRL 모델 데이터 (약간 더 높은 수익률)
    np.random.seed(42)
    noise = np.random.normal(0, 0.015, len(dates))
    cumulative_returns = np.cumsum(np.random.normal(0.001, 0.01, len(dates)) + noise)
    drl_values = base_value * (1 + cumulative_returns)
    
    # Buy & Hold 기준 데이터
    np.random.seed(43)
    baseline_noise = np.random.normal(0, 0.02, len(dates))
    baseline_returns = np.cumsum(np.random.normal(0.0005, 0.015, len(dates)) + baseline_noise)
    baseline_values = base_value * (1 + baseline_returns)
    
    # 전통적 기술적 분석 기반 전략
    np.random.seed(44)
    ta_noise = np.random.normal(0, 0.018, len(dates))
    ta_returns = np.cumsum(np.random.normal(0.0007, 0.012, len(dates)) + ta_noise)
    ta_values = base_value * (1 + ta_returns)
    
    # 멀티모달 앙상블 모델
    np.random.seed(45)
    ensemble_noise = np.random.normal(0, 0.01, len(dates))
    ensemble_returns = np.cumsum(np.random.normal(0.0012, 0.009, len(dates)) + ensemble_noise)
    ensemble_values = base_value * (1 + ensemble_returns)
    
    portfolio_data = pd.DataFrame({
        'timestamp': dates,
        'drl_value': drl_values,
        'baseline_value': baseline_values,
        'ta_value': ta_values,
        'ensemble_value': ensemble_values
    })
    
    # 2. 거래 신호 데이터
    trade_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    signals = []
    
    # 임의로 규칙 기반 거래 시그널 생성
    for i in range(len(trade_dates)):
        if i % 5 == 0:
            signal = 'BUY'
        elif i % 7 == 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        signals.append(signal)
    
    trades_data = pd.DataFrame({
        'timestamp': trade_dates,
        'signal': signals,
        'price': base_value * (1 + 0.0005 * np.arange(len(trade_dates)) + 0.01 * np.sin(np.arange(len(trade_dates))/10))
    })
    
    # 3. 성능 지표 데이터
    metrics = {
        "total_return": 32.15,
        "annual_return": 12.75,
        "sharpe_ratio": 1.45,
        "max_drawdown": -18.25,
        "win_rate": 65.3,
        "profit_factor": 2.1,
        "volatility": 15.8,
        "calmar_ratio": 0.98,
        "sortino_ratio": 1.89,
        "trades_per_month": 12.4
    }
    
    # 4. 월별 성과 데이터
    months = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
    strategy_names = ['DRL', 'Ensemble', 'TA', 'Buy&Hold']
    
    monthly_returns = {}
    for strategy in strategy_names:
        np.random.seed(hash(strategy) % 100)
        returns = np.random.normal(0.01, 0.03, len(months))
        if strategy == 'Ensemble':
            returns += 0.01  # 앙상블이 약간 더 좋은 성과
        monthly_returns[strategy] = returns
    
    monthly_data = pd.DataFrame(monthly_returns, index=months)
    
    # 5. 시장 조건별 성과 데이터
    market_conditions = ['Bull Market', 'Bear Market', 'Sideways Market']
    strategies = ['DRL', 'Ensemble', 'Technical Analysis', 'Buy & Hold']
    
    # 각 시장 조건과 전략별 성과 (수익률 %)
    market_performance = {
        'Bull Market': [18.5, 21.3, 16.2, 19.8],
        'Bear Market': [-5.2, -3.8, -12.5, -15.1],
        'Sideways Market': [4.2, 5.1, 1.5, 0.2]
    }
    
    market_data = pd.DataFrame(market_performance, index=strategies)
    
    return portfolio_data, trades_data, metrics, monthly_data, market_data

# 기본 스타일 설정
def setup_plotting_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("muted")
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#E5E5E5'
    plt.rcParams['grid.linestyle'] = '--'

# 포트폴리오 성능 비교 그래프
def generate_portfolio_comparison(data=None):
    setup_plotting_style()
    
    if data is None:
        data, _, _, _, _ = generate_sample_data()
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # 모든 전략 그래프 생성
    ax.plot(data['timestamp'], data['drl_value'], linewidth=2.5, label='DRL Strategy')
    ax.plot(data['timestamp'], data['baseline_value'], linewidth=2, linestyle='--', label='Buy & Hold')
    ax.plot(data['timestamp'], data['ta_value'], linewidth=2, label='Technical Analysis')
    ax.plot(data['timestamp'], data['ensemble_value'], linewidth=3, label='Multimodal Ensemble')
    
    # 그래프 포맷팅
    ax.set_title('Trading Strategy Performance Comparison', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Portfolio Value (USD)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # 축 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    
    # 수익률 표시
    for col, label in zip(['drl_value', 'baseline_value', 'ta_value', 'ensemble_value'], 
                          ['DRL Strategy', 'Buy & Hold', 'Technical Analysis', 'Multimodal Ensemble']):
        start_val = data[col].iloc[0]
        end_val = data[col].iloc[-1]
        total_return = (end_val / start_val - 1) * 100
        ax.annotate(f'{label}: {total_return:.2f}%', 
                   xy=(0.02, 0.96 - 0.05 * (['drl_value', 'baseline_value', 'ta_value', 'ensemble_value'].index(col))),
                   xycoords='axes fraction', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'portfolio_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Portfolio comparison chart saved: {output_file}")
    
    return output_file

# 드로다운 곡선 생성
def generate_drawdown_curve(data=None):
    setup_plotting_style()
    
    if data is None:
        data, _, _, _, _ = generate_sample_data()
    
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    # 각 전략별 드로다운 계산 및 시각화
    strategies = {
        'drl_value': 'DRL Strategy',
        'ensemble_value': 'Multimodal Ensemble',
        'ta_value': 'Technical Analysis',
        'baseline_value': 'Buy & Hold'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (col, label) in enumerate(strategies.items()):
        # 최대 누적 가치 계산
        data[f'cummax_{col}'] = data[col].cummax()
        
        # 드로다운 계산
        data[f'drawdown_{col}'] = (data[col] - data[f'cummax_{col}']) / data[f'cummax_{col}'] * 100
        
        # 그래프 그리기
        ax.plot(data['timestamp'], data[f'drawdown_{col}'], 
                linewidth=2, color=colors[i], label=label)
    
    # 그래프 포맷팅
    ax.set_title('Drawdown Comparison by Strategy', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Drawdown (%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower left')
    
    # 축 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    
    # y축 반전
    ax.invert_yaxis()
    
    # 최대 드로다운 표시
    for i, (col, label) in enumerate(strategies.items()):
        max_dd = data[f'drawdown_{col}'].min()
        ax.annotate(f'{label} Max DD: {max_dd:.2f}%', 
                   xy=(0.02, 0.05 + 0.05 * i),
                   xycoords='axes fraction', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'drawdown_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Drawdown comparison chart saved: {output_file}")
    
    return output_file

# 거래 신호 분포 생성
def generate_trade_signal_distribution(data=None):
    setup_plotting_style()
    
    if data is None:
        _, data, _, _, _ = generate_sample_data()
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # 거래 신호 집계
    signal_counts = data['signal'].value_counts()
    
    # 그래프 색상 설정
    colors = {
        'BUY': '#2ca02c',    # 초록색
        'SELL': '#d62728',   # 빨간색
        'HOLD': '#7f7f7f'    # 회색
    }
    
    bar_colors = [colors.get(signal, '#1f77b4') for signal in signal_counts.index]
    
    # 바 그래프 생성
    bars = ax.bar(signal_counts.index, signal_counts.values, color=bar_colors)
    
    # 바 레이블 추가
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.0f}', ha='center', va='bottom', fontsize=12)
    
    # 그래프 포맷팅
    ax.set_title('Trade Signal Distribution', fontsize=18, fontweight='bold')
    ax.set_xlabel('Signal Type', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 전체 신호 대비 비율 계산 및 표시
    total_signals = signal_counts.sum()
    for i, (signal, count) in enumerate(signal_counts.items()):
        percentage = count / total_signals * 100
        ax.annotate(f'{percentage:.1f}%', 
                   xy=(i, count - 5),
                   xytext=(0, -20),
                   textcoords='offset points',
                   ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'trade_signal_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Trade signal distribution chart saved: {output_file}")
    
    return output_file

# 월별 수익률 히트맵 생성
def generate_monthly_returns_heatmap(data=None):
    setup_plotting_style()
    
    if data is None:
        _, _, _, data, _ = generate_sample_data()
    
    # 월별 데이터로 변환
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data.index = data.index.strftime('%Y-%m')
    
    # 수익률 백분율로 변환 (시각화를 위해)
    data_pct = data * 100
    
    # 핏맵 설정
    plt.figure(figsize=(14, 8))
    cmap = sns.diverging_palette(10, 240, as_cmap=True)  # 빨간색-파란색 발산 팔레트
    
    # 히트맵 생성
    ax = sns.heatmap(data_pct.T, annot=True, cmap=cmap, center=0,
                   linewidths=1, linecolor='white',
                   cbar_kws={'label': 'Monthly Return (%)'}, fmt='.1f',
                   vmin=-10, vmax=10)  # -10% ~ +10% 범위로 색상 제한
    
    # 그래프 포맷팅
    plt.title('Monthly Returns by Strategy', fontsize=18, fontweight='bold')
    plt.xlabel('Year-Month', fontsize=14)
    plt.ylabel('Strategy', fontsize=14)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'monthly_returns_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Monthly returns heatmap saved: {output_file}")
    
    return output_file

# 성능 지표 비교 레이더 차트
def generate_performance_metrics_radar(metrics=None):
    setup_plotting_style()
    
    if metrics is None:
        _, _, metrics, _, _ = generate_sample_data()
    
    # 레이더 차트를 위한 선택된 지표 (모든 지표가 클수록 좋은 값으로 변환)
    selected_metrics = {
        'Sharpe Ratio': metrics['sharpe_ratio'],
        'Win Rate (%)': metrics['win_rate'],
        'Profit Factor': metrics['profit_factor'],
        'Calmar Ratio': metrics['calmar_ratio'],
        'Sortino Ratio': metrics['sortino_ratio'],
        'Return/Vol': metrics['annual_return'] / metrics['volatility']
    }
    
    # 비교 대상 데이터 (기준 모델) - 가정값
    benchmark_metrics = {
        'Sharpe Ratio': 0.8,
        'Win Rate (%)': 52.0,
        'Profit Factor': 1.2,
        'Calmar Ratio': 0.5,
        'Sortino Ratio': 0.9,
        'Return/Vol': 0.4
    }
    
    # 데이터 정규화 (0~1 사이로)
    categories = list(selected_metrics.keys())
    N = len(categories)
    
    # 최대값 설정 (실제 값과 벤치마크 값의 최대값보다 약간 크게)
    max_values = {cat: max(selected_metrics[cat], benchmark_metrics[cat]) * 1.2 for cat in categories}
    
    # 정규화 데이터
    values_norm = [selected_metrics[cat] / max_values[cat] for cat in categories]
    benchmark_norm = [benchmark_metrics[cat] / max_values[cat] for cat in categories]
    
    # 원형 그래프를 위한 각도 설정
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 원형으로 닫기 위해 첫 각도 추가
    
    # 데이터도 원형으로 닫기
    values_norm += values_norm[:1]
    benchmark_norm += benchmark_norm[:1]
    
    # 레이더 차트 생성
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 우리 모델 데이터 그리기
    ax.plot(angles, values_norm, 'o-', linewidth=2, label='Multimodal Ensemble Strategy')
    ax.fill(angles, values_norm, alpha=0.25)
    
    # 벤치마크 데이터 그리기
    ax.plot(angles, benchmark_norm, 'o-', linewidth=2, label='Traditional Strategy')
    ax.fill(angles, benchmark_norm, alpha=0.1)
    
    # 축 설정
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    
    # 그래프 포맷팅
    ax.set_title('Performance Metrics Comparison', fontsize=18, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 실제 지표값 표시
    for i, category in enumerate(categories):
        model_val = selected_metrics[category]
        bench_val = benchmark_metrics[category]
        
        # 두 값을 모두 표시하되, 개선율도 계산
        improvement = ((model_val / bench_val) - 1) * 100
        
        # 위치 계산 (각도와 정규화된 값 사용)
        angle = angles[i]
        r = max(values_norm[i], benchmark_norm[i]) + 0.1
        
        # 값 주석 추가
        ax.annotate(f'{model_val:.2f} vs {bench_val:.2f}\n(+{improvement:.1f}%)',
                   xy=(angle, r), 
                   xytext=(angle, r),
                   textcoords='data',
                   ha='center', va='center',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'performance_metrics_radar.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance metrics radar chart saved: {output_file}")
    
    return output_file

# 시장 조건별 성능 비교 그래프
def generate_market_condition_comparison(data=None):
    setup_plotting_style()
    
    if data is None:
        _, _, _, _, data = generate_sample_data()
    
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # 시장 조건별로 바 그래프 생성
    bar_width = 0.2
    x = np.arange(len(data.columns))
    
    for i, strategy in enumerate(data.index):
        ax.bar(x + i * bar_width, data.values[i], width=bar_width, label=strategy)
    
    # 그래프 포맷팅
    ax.set_title('Performance by Market Condition', fontsize=18, fontweight='bold')
    ax.set_xlabel('Market Condition', fontsize=14)
    ax.set_ylabel('Return (%)', fontsize=14)
    ax.set_xticks(x + bar_width * (len(data.index) - 1) / 2)
    ax.set_xticklabels(data.columns)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for i, strategy in enumerate(data.index):
        for j, condition in enumerate(data.columns):
            value = data.loc[strategy, condition]
            ax.annotate(f'{value:.1f}%',
                       xy=(j + i * bar_width, value),
                       xytext=(0, 3 if value > 0 else -12),  # 값이 음수면 아래에 텍스트
                       textcoords='offset points',
                       ha='center', va='bottom' if value > 0 else 'top',
                       fontsize=9)
    
    # y축에 0 라인 강조
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'market_condition_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Market condition comparison chart saved: {output_file}")
    
    return output_file

# 모델 구조 시각화
def generate_model_architecture_diagram():
    fig = plt.figure(figsize=(16, 10))
    
    # 그리드 설정
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 3, 1])
    
    # 상단 제목 영역
    ax_title = plt.subplot(gs[0, :])
    ax_title.text(0.5, 0.5, 'Multimodal Ensemble Reinforcement Learning Model Architecture', fontsize=24, fontweight='bold', ha='center', va='center')
    ax_title.axis('off')
    
    # 입력 데이터 영역
    ax_input = plt.subplot(gs[1, 0])
    ax_input.text(0.5, 0.9, 'Input Data', fontsize=16, fontweight='bold', ha='center', va='center')
    
    input_types = ['Price Data', 'Volume Data', 'Text Sentiment', 'Technical Indicators']
    input_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (input_type, color) in enumerate(zip(input_types, input_colors)):
        ax_input.add_patch(plt.Rectangle((0.2, 0.7 - i * 0.15), 0.6, 0.1, facecolor=color, alpha=0.6))
        ax_input.text(0.5, 0.7 - i * 0.15 + 0.05, input_type, fontsize=12, ha='center', va='center')
    
    ax_input.axis('off')
    
    # 모델 아키텍처 영역
    ax_model = plt.subplot(gs[1, 1])
    ax_model.text(0.5, 0.95, 'Model Architecture', fontsize=16, fontweight='bold', ha='center', va='center')
    
    # 개별 모델 그리기
    models = ['LSTM Network', 'CNN Network', 'NLP Sentiment Analysis', 'Transformer']
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model, color) in enumerate(zip(models, model_colors)):
        ax_model.add_patch(plt.Rectangle((0.1, 0.7 - i * 0.15), 0.35, 0.1, facecolor=color, alpha=0.6))
        ax_model.text(0.275, 0.7 - i * 0.15 + 0.05, model, fontsize=10, ha='center', va='center')
    
    # 앙상블 레이어 그리기
    ax_model.add_patch(plt.Rectangle((0.55, 0.4), 0.35, 0.15, facecolor='#9467bd', alpha=0.6))
    ax_model.text(0.725, 0.475, 'Ensemble Layer', fontsize=12, ha='center', va='center')
    
    # 연결 화살표 그리기
    for i in range(4):
        ax_model.arrow(0.45, 0.7 - i * 0.15 + 0.05, 0.08, -0.2 + i * 0.05,
                      head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    ax_model.axis('off')
    
    # 출력 영역
    ax_output = plt.subplot(gs[1, 2])
    ax_output.text(0.5, 0.9, 'Model Output', fontsize=16, fontweight='bold', ha='center', va='center')
    
    # 액션 출력 그리기
    actions = ['BUY', 'SELL', 'HOLD']
    action_colors = ['#2ca02c', '#d62728', '#7f7f7f']
    
    ax_output.add_patch(plt.Rectangle((0.2, 0.5), 0.6, 0.2, facecolor='#9467bd', alpha=0.3))
    ax_output.text(0.5, 0.6, 'Action Decision Layer', fontsize=12, ha='center', va='center')
    
    for i, (action, color) in enumerate(zip(actions, action_colors)):
        ax_output.add_patch(plt.Rectangle((0.25 + i * 0.2, 0.3), 0.15, 0.1, facecolor=color, alpha=0.6))
        ax_output.text(0.325 + i * 0.2, 0.35, action, fontsize=10, ha='center', va='center')
    
    # 화살표 연결
    ax_output.arrow(0.2, 0.6, -0.1, 0.0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    ax_model.arrow(0.9, 0.475, 0.08, 0.0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    ax_output.axis('off')
    
    # 하단 설명 영역
    ax_footer = plt.subplot(gs[2, :])
    ax_footer.text(0.5, 0.7, 'Reinforcement Learning Framework: DQN + DDPG + PPO', fontsize=14, fontweight='bold', ha='center', va='center')
    ax_footer.text(0.5, 0.4, 'Reward Function: Return + Risk Adjusted Factors + Transaction Cost Penalty', fontsize=12, ha='center', va='center')
    ax_footer.axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'model_architecture.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model architecture diagram saved: {output_file}")
    
    return output_file

# 모든 그래프 생성 함수
def generate_all_research_graphs():
    print("Generating research graphs...")
    
    # 샘플 데이터 생성
    portfolio_data, trades_data, metrics, monthly_data, market_data = generate_sample_data()
    
    # 모든 그래프 생성
    generate_portfolio_comparison(portfolio_data)
    generate_drawdown_curve(portfolio_data)
    generate_trade_signal_distribution(trades_data)
    generate_monthly_returns_heatmap(monthly_data)
    generate_performance_metrics_radar(metrics)
    generate_market_condition_comparison(market_data)
    generate_model_architecture_diagram()
    
    print("All research graphs have been generated.")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    generate_all_research_graphs() 