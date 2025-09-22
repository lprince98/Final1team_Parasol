import streamlit as st
from PIL import Image # Pillow 라이브러리 import
import pandas as pd
import plotly.graph_objects as go
import random # 예시 데이터 생성을 위해 사용

# --- AWS 연동을 위한 준비 (실제로는 이 부분에 boto3 클라이언트 설정) ---
# import boto3
# dynamodb = boto3.resource('dynamodb')
# bedrock = boto3.client('bedrock-runtime')
# --------------------------------------------------------------------

def get_report_data(user_id: str, test_id: str):
    """
    [함수 설명]
    실제 서비스에서는 이 함수가 AWS DynamoDB와 Bedrock에 연결하여
    사용자의 분석 결과 및 LLM이 생성한 보고서 텍스트를 가져옵니다.
    이 예제에서는 데모를 위해 무작위 샘플 데이터를 생성합니다.
    """
    # ----- ▼▼▼ 이 부분 전체가 나중에 AWS 연동 코드로 대체됩니다 ▼▼▼ -----
    
    # 샘플 데이터 생성
    risk_levels = ["안정", "경계", "상담 권장"]
    chosen_risk = random.choice(risk_levels)
    
    score = 0
    if chosen_risk == "안정":
        score = random.randint(20, 49)
    elif chosen_risk == "경계":
        score = random.randint(50, 79)
    else:
        score = random.randint(80, 99)

    # LLM이 생성했다고 가정한 텍스트들
    summary_text = {
        "안정": "이번 검사에서 이성현님의 움직임 패턴은 전형적인 파킨슨병의 패턴과 높은 유사도를 보이며 안정적인 범위에 있습니다. 꾸준한 자기 관리를 통해 현재 상태를 잘 유지하시는 것이 중요합니다.",
        "경계": "이번 검사에서 이성현님의 움직임 패턴이 전형적인 파킨슨병의 패턴과 다소 다른 몇 가지 특징을 보였습니다. 큰 우려를 할 단계는 아니지만, 주기적인 검사를 통해 변화를 관찰하는 것이 좋습니다.",
        "상담 권장": "전형적인 파킨슨병의 운동 패턴과 다른 뚜렷한 특징이 관찰되었습니다. 이 결과가 확정적인 진단은 아니며, 보다 정확한 상태 파악을 위해 신경과 전문의와 상담해 보시는 것을 강력히 권장합니다."
    }
    
    tapping_text = {
        "안정": "핑거 태핑의 속도와 리듬이 매우 일정하게 유지되었습니다. 움직임의 진폭 또한 꾸준하여 안정적인 제어 능력을 보여줍니다.",
        "경계": "핑거 태핑 후반부로 갈수록 움직임의 폭이 약간 줄어드는 '진폭 감소' 현상이 관찰되었습니다. 이는 파킨슨 증후군의 일반적인 특징 중 하나입니다.",
        "상담 권장": "핑거 태핑의 리듬이 다소 불규칙하고, 움직임의 진폭이 뚜렷하게 감소하는 패턴이 나타났습니다. 이는 움직임 조절 능력의 변화를 시사할 수 있습니다."
    }
        
    eye_text = {
        "안정": "핑거 태핑의 속도와 리듬이 매우 일정하게 유지되었습니다. 움직임의 진폭 또한 꾸준하여 안정적인 제어 능력을 보여줍니다.",
        "경계": "핑거 태핑 후반부로 갈수록 움직임의 폭이 약간 줄어드는 '진폭 감소' 현상이 관찰되었습니다. 이는 파킨슨 증후군의 일반적인 특징 중 하나입니다.",
        "상담 권장": "핑거 태핑의 리듬이 다소 불규칙하고, 움직임의 진폭이 뚜렷하게 감소하는 패턴이 나타났습니다. 이는 움직임 조절 능력의 변화를 시사할 수 있습니다."
    }

    # 최종적으로 서버가 반환할 데이터 형식
    return {
        "userName": "이성현",
        "testDate": "2025년 9월 11일",
        "riskLevel": chosen_risk,
        "reportText": {
            "summary": summary_text[chosen_risk],
            "eye": eye_text[chosen_risk],
            "finger_tapping": tapping_text[chosen_risk],
            "voice": "음성 분석 결과, 목소리의 떨림이나 끊김은 정상 범위에 있습니다." # 예시
        },
        "scores": {
            "atypicalityScore": score
        }
    }
    # ----- ▲▲▲ 여기까지가 AWS 연동 코드로 대체될 부분 ▲▲▲ -----


# --- Streamlit UI 구성 ---

# 1. 페이지 기본 설정
# 아이콘 이미지 로드
icon = Image.open("parasol.png") # 파일 이름은 실제 파일명으로 변경

# 페이지 설정에서 page_icon으로 이미지 객체 전달
st.set_page_config(
    page_title="파라솔 리포트", 
    page_icon=icon, # "☂️" 대신 이미지 객체 사용
    layout="wide"
)

# 2. 데이터 로드
# URL 쿼리 파라미터나 로그인 정보에서 userId, testId를 가져올 수 있습니다.
# 예시: http://localhost:8501?user_id=user123&test_id=test_abc
user_id_from_query = st.query_params.get("user_id", "user123") # 기본값
test_id_from_query = st.query_params.get("test_id", "test_abc") # 기본값
report_data = get_report_data(user_id_from_query, test_id_from_query)


# 3. 헤더
# st.columns를 사용하여 로고와 제목을 위한 공간을 나눔 (예: 1:5 비율)
col1, col2 = st.columns([1, 7])

with col1:
    # 1번 컬럼(왼쪽)에 로고 이미지 표시
    st.image(icon, width=200) # 로고 파일 경로, 너비 조절

with col2:
    # 2번 컬럼(오른쪽)에 제목과 헤더 표시 (이모지 제거)
    st.title("'파라솔' 나의 움직임 건강 리포트")
    st.header(f"{report_data['userName']}님을 위한 분석 결과입니다.")

# 캡션과 구분선은 컬럼 밖에서 전체 너비로 표시
st.caption(f"검사 일시: {report_data['testDate']}")
st.divider()


# 4. 종합 소견 (Summary)
risk_level = report_data['riskLevel']
if risk_level == "상담 권장":
    st.subheader("🔴 전문의와의 상담을 권장합니다")
elif risk_level == "경계":
    st.subheader("🟡 주의 깊은 관찰이 필요해요")
else:
    st.subheader("🟢 안정적인 패턴을 보이고 있어요")
st.write(report_data['reportText']['summary'])
st.divider()


# 5. 세부 분석 결과
st.header("세부 분석 결과")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👁️ 시선 분석")
    st.info(report_data['reportText']['eye'])

with col2:
    st.subheader("🖐️ 손가락 움직임 분석")
    st.info(report_data['reportText']['finger_tapping'])
    score = report_data['scores']['atypicalityScore']
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "비정형 지수"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"}, # 게이지 바는 투명하게
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 80], 'color': 'lightyellow'},
                {'range': [80, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }))
    st.plotly_chart(fig, use_container_width=True)
    

with col3:
    st.subheader("🎤 목소리 분석")
    st.info(report_data['reportText']['voice'])
    # (여기에 목소리 관련 그래프 추가)
st.divider()

# 6. 다음 단계 안내
st.header("다음 단계 안내")
st.success("""
**이 결과를 바탕으로 무엇을 할 수 있을까요?**

- **리포트 저장 및 출력:** 병원 방문 시 의사 선생님께 보여드리면 진료에 큰 도움이 됩니다.
- **전문의 상담:** 정확한 진단은 반드시 신경과 전문의와의 대면 진료를 통해 이루어져야 합니다.
- **주기적인 검사:** 3개월 뒤에 다시 검사하여 움직임 패턴의 변화를 추적해 보세요. '파라솔'이 함께하겠습니다.
""")

# 7. 법적 고지
st.warning("""
**법적 고지:** 본 리포트는 '파라솔'의 AI 분석 모델이 제공하는 참고 정보이며, 의학적 진단이 아닙니다. 모든 의학적 판단과 치료는 반드시 신경과 전문의와의 대면 진료를 통해 이루어져야 합니다.
""")