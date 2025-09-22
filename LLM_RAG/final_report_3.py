import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
from PIL import Image
from scipy.signal import find_peaks
from scipy.stats import linregress

# --- 1. 그래프 생성 함수들 ---

def create_gaze_polar_chart(gaze_data):
    """시선 추적 분석용 폴라 차트 생성 함수"""
    theta = ['상(Up)', '우(Right)', '하(Down)', '좌(Left)', '상(Up)']
    normal_range = [85, 95, 85, 95, 85]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normal_range, theta=theta, fill='toself', fillcolor='rgba(0, 176, 246, 0.2)',
        line_color='rgba(0, 176, 246, 0.5)', name='정상 범위'
    ))
    fig.add_trace(go.Scatterpolar(
        r=gaze_data + [gaze_data[0]], theta=theta, fill='none',
        line=dict(color='red', width=3), name='사용자 측정값'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="방향별 안구 운동 능력", showlegend=True, height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_voice_radar_chart(user_voice_data):
    """음성 분석용 레이더 차트 생성 함수"""
    categories = ['음성 떨림', '음높이 안정성', '목소리 선명도', '목소리 크기']
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=user_voice_data + [user_voice_data[0]], theta=categories + [categories[0]],
        fill='toself', name='사용자 음성 패턴', line_color='red'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="음성 시그니처 분석", height=350,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def create_tapping_timeseries_chart(tapping_df):
    """핑거 태핑 분석용 시계열 차트 생성 함수"""
    time = tapping_df['time']
    tapping_data = tapping_df['angle']
    
    peaks, _ = find_peaks(tapping_data, height=np.mean(tapping_data), distance=15)
    peak_times = time.iloc[peaks]
    peak_amps = tapping_data.iloc[peaks]

    slope, intercept = 0, 0
    if len(peak_times) > 1:
        slope, intercept, _, _, _ = linregress(peak_times, peak_amps)
        trendline = slope * time + intercept
    else:
        trendline = np.full_like(time, np.mean(tapping_data))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=tapping_data, mode='lines', name='핑거 태핑 각도'))
    fig.add_trace(go.Scatter(x=peak_times, y=peak_amps, mode='markers', name='피크 지점', marker=dict(color='red', size=8)))
    fig.add_trace(go.Scatter(x=time, y=trendline, mode='lines', name='진폭 감소 추세선', line=dict(color='black', dash='dash')))
    fig.update_layout(
        title=f"시간에 따른 움직임 변화 (추세선 기울기: {slope:.2f})",
        xaxis_title="시간 (초)", yaxis_title="손가락 각도 (도)", showlegend=True
    )
    return fig

# --- 2. 데이터 로드 함수 (모든 예시 데이터를 생성하도록 확장) ---
def get_report_data(user_id: str, test_id: str):
    risk_level = "상담 권장" # 테스트를 위해 특정 값으로 고정
    score = 87
    summary_text = "전형적인 파킨슨병의 운동 패턴과 다른 뚜렷한 특징이 관찰되었습니다. 이는 움직임 조절 능력의 변화를 시사할 수 있습니다. 이 결과가 확정적인 진단은 아니며, 보다 정확한 상태 파악을 위해 신경과 전문의와 상담해 보시는 것을 강력히 권장합니다."
    
    return {
        "userName": "이성현", "testDate": "2025년 9월 12일", "riskLevel": risk_level,
        "reportText": {"summary": summary_text},
        "scores": {"atypicalityScore": score},
        "gaze_data": [30, 90, 25, 92],
        "voice_data": [80, 50, 30, 50],
        "tapping_data": pd.DataFrame({
            'time': np.linspace(0, 15, 450),
            'angle': 20 * np.sin(np.linspace(0, 15, 450) * 2 * np.pi * 0.7) * np.linspace(1, 0.4, 450) + 60 + np.random.normal(0, 1.5, 450)
        })
    }

# --- 3. Streamlit UI 구성 ---

# 페이지 기본 설정
try:
    icon = Image.open("parasol.png")
except FileNotFoundError:
    icon = "☂️"
st.set_page_config(page_title="파라솔 리포트", page_icon=icon, layout="wide")

# 데이터 로드
report_data = get_report_data("user123", "test_abc")

# 헤더
col1, col2 = st.columns([1, 7])
with col1:
    if isinstance(icon, str):
        st.title(icon)
    else:
        st.image(icon, width=180)
with col2:
    st.title("'파라솔' 나의 움직임 건강 리포트")
    st.header(f"{report_data['userName']}님을 위한 분석 결과입니다.")
st.caption(f"검사 일시: {report_data['testDate']}")
st.divider()

# 메인 대시보드 레이아웃
main_col, side_col = st.columns([2, 1])

with main_col:
    # 종합 소견 카드
    with st.container(border=True):
        st.subheader("🔴 전문의와의 상담을 권장합니다")
        st.write(report_data['reportText']['summary'])

    # 핑거 태핑 상세 분석 그래프
    with st.container(border=True):
        st.subheader("🖐️ 손가락 움직임 분석 (정상 vs PD)")
        tapping_fig = create_tapping_timeseries_chart(report_data['tapping_data'])
        st.plotly_chart(tapping_fig, use_container_width=True)
        
    # with st.container(border=True):
    #     st.subheader("🖐️ 손가락 움직임 분석 (정상 vs PD)")
    #     tapping_fig = create_tapping_timeseries_chart(report_data['tapping_data'])
    #     st.plotly_chart(tapping_fig, use_container_width=True)

with side_col:
    # 시선 추적 분석 카드
    with st.container(border=True):
        st.subheader("👁️ 시선 추적 분석 (정상 vs PSP)")
        gaze_fig = create_gaze_polar_chart(report_data['gaze_data'])
        st.plotly_chart(gaze_fig, use_container_width=True)

    # 음성 분석 카드
    with st.container(border=True):
        st.subheader("🎤 음성 분석 (다중 질환 비교)")
        voice_fig = create_voice_radar_chart(report_data['voice_data'])
        st.plotly_chart(voice_fig, use_container_width=True)

# 법적 고지
st.warning("법적 고지: 본 리포트는 의료적 진단이 아니며, 참고용으로만 사용해야 합니다. 모든 의학적 판단과 치료는 반드시 신경과 전문의와의 대면 진료를 통해 이루어져야 합니다.")