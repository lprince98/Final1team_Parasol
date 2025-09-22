import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import random
from PIL import Image

# ----------------- 데이터 로드 함수 (이전과 동일) -----------------
def get_report_data(user_id: str, test_id: str):
    # 이 부분은 실제 AWS DynamoDB/Bedrock에서 데이터를 가져오는 로직으로 대체됩니다.
    risk_levels = ["안정", "경계", "상담 권장"]
    chosen_risk = random.choice(risk_levels)
    score = 0
    if chosen_risk == "안정": score = random.randint(20, 49)
    elif chosen_risk == "경계": score = random.randint(50, 79)
    else: score = random.randint(80, 99)
    
    summary_text = {
        "안정": "이번 검사에서 이성현님의 움직임 패턴은 전형적인 파킨슨병의 패턴과 높은 유사도를 보이며 안정적인 범위에 있습니다. 꾸준한 자기 관리를 통해 현재 상태를 잘 유지하시는 것이 중요합니다.",
        "경계": "이번 검사에서 이성현님의 움직임 패턴이 전형적인 파킨슨병의 패턴과 다소 다른 몇 가지 특징을 보였습니다. 큰 우려를 할 단계는 아니지만, 주기적인 검사를 통해 변화를 관찰하는 것이 좋습니다.",
        "상담 권장": "전형적인 파킨슨병의 운동 패턴과 다른 뚜렷한 특징이 관찰되었습니다. 이 결과가 확정적인 진단은 아니며, 보다 정확한 상태 파악을 위해 신경과 전문의와 상담해 보시는 것을 강력히 권장합니다."
    }
    
    return {
        "userName": "이성현", "testDate": "2025년 9월 12일", "riskLevel": chosen_risk,
        "reportText": {"summary": summary_text[chosen_risk]},
        "scores": {
            "atypicalityScore": score,
            "amplitudeDecrement": -0.25 if chosen_risk == "안정" else -0.65,
            "rhythmVariability": 0.32 if chosen_risk == "안정" else 0.88
        },
        "time_series_data": pd.DataFrame({ # 상세 분석 그래프를 위한 시계열 데이터 예시
            'time': np.linspace(0, 15, 300),
            'angle': 60 + 20 * np.sin(np.linspace(0, 15, 300) * 2 * np.pi) + np.random.randn(300) * 2
        })
    }
# ------------------------------------------------------------------------------------


# --- Streamlit UI 구성 ---

# 1. 페이지 기본 설정
icon = Image.open("parasol.png") # 로고 파일이 있다고 가정
st.set_page_config(page_title="파라솔 리포트", page_icon=icon, layout="wide")

# 2. 데이터 로드
report_data = get_report_data("user123", "test_abc")

# 3. 헤더
col1, col2 = st.columns([1, 5])
with col1:
    st.image(icon, width=100)
with col2:
    st.title("'파라솔' 운동 패턴 분석 리포트")
    st.header(f"{report_data['userName']}님을 위한 분석 결과입니다.")
st.caption(f"검사 일시: {report_data['testDate']}")
st.divider()

# 4. 메인 대시보드 레이아웃 (2:1 비율로 컬럼 나누기)
main_col, side_col = st.columns([2, 1])

# --- 왼쪽 메인 컬럼 ---
with main_col:
    # 4-1. 종합 소견 카드
    with st.container(border=True):
        risk_level = report_data['riskLevel']
        if risk_level == "상담 권장":
            st.subheader("🔴 전문의와의 상담을 권장합니다")
        elif risk_level == "경계":
            st.subheader("🟡 주의 깊은 관찰이 필요해요")
        else:
            st.subheader("🟢 안정적인 패턴을 보이고 있어요")
        st.write(report_data['reportText']['summary'])

    # 4-2. 운동 패턴 상세 분석 그래프
    with st.container(border=True):
        st.subheader("📈 운동 패턴 상세 분석")
        df = report_data['time_series_data']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['time'], y=df['angle'], mode='lines', name='핑거 태핑 각도'))
        fig.update_layout(title='시간에 따른 움직임 변화', xaxis_title='시간 (초)', yaxis_title='손가락 각도 (도)')
        st.plotly_chart(fig, use_container_width=True)


# --- 오른쪽 사이드바 컬럼 ---
with side_col:
    # 4-3. 상세 지표 카드
    with st.container(border=True):
        st.subheader("📊 상세 지표")
        # st.metric을 사용하여 핵심 지표 강조
        st.metric(label="비정형 지수", value=f"{report_data['scores']['atypicalityScore']} 점")
        st.metric(label="진폭 감소율", value=f"{report_data['scores']['amplitudeDecrement']:.2f}")
        st.metric(label="리듬 변동성", value=f"{report_data['scores']['rhythmVariability']:.2f}")

    # 4-4. 다음 단계 안내 카드
    with st.container(border=True):
        st.subheader("➡️ 다음 단계 안내")
        st.info("""
        - **리포트 저장:** 이 결과를 저장하여 의사 선생님과 상담 시 활용하세요.
        - **전문의 상담:** 정확한 진단은 반드시 전문의를 통해 받아야 합니다.
        - **주기적인 검사:** 3개월 뒤 다시 검사하여 변화를 추적해 보세요.
        """)

# 5. 법적 고지
st.warning("법적 고지: 본 리포트는 의료적 진단이 아니며, 참고용으로만 사용해야 합니다. 모든 의학적 판단과 치료는 반드시 신경과 전문의와의 대면 진료를 통해 이루어져야 합니다.")