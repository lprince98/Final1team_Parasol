# -*- coding: utf-8 -*-
import streamlit as st
import json
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import os

# =========================
# .env 로드 (API 키)
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY를 불러오지 못했습니다. .env 파일을 확인하세요.")
client = OpenAI(api_key=api_key)

# =========================
# 페이지 설정
# =========================
st.set_page_config(layout="centered", page_title="파라솔 검사 결과")

# =========================
# 샘플 JSON 데이터 (실제 상황에서는 모델 출력 JSON 불러오기)
# =========================
data = {
    "comprehensive_results": {
        "group_probabilities": {
            "HC": 10,
            "PD": 15,
            "PSP": 18,
            "MSA": 57
        },
        "final_diagnosis": "MSA (다계통위축증)",
        "summary": {
            "executive_summary": "MSA의 특징적인 양상이 두드러지게 나타났습니다.",
            "recommendations": "전문의 상담을 통한 영상검사 및 약물 조정이 필요합니다.",
            "next_steps": "가족의 생활 지원 준비, 정기적 신경과 진료, 재활 치료 병행"
        }
    }
}

# =========================
# 데이터 파싱
# =========================
probs = data["comprehensive_results"]["group_probabilities"]
final_diagnosis = data["comprehensive_results"]["final_diagnosis"]

# =========================
# 스타일 정의
# =========================
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
    background-color: #2C3E91;
    padding: 10px;
    border-radius: 8px;
}
.report-box {
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: white;
    background-color: #E74C3C;
    padding: 12px;
    border-radius: 10px;
    margin: 10px 0;
}
.card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
    font-size: 22px;   /* ✅ 라벨 + 숫자 동일 크기 */
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 상단 제목
# =========================
st.markdown('<div class="title">검사 결과</div>', unsafe_allow_html=True)

# 위험도 보고서
st.markdown(
    f'<div class="report-box">당신은 <b>{final_diagnosis}</b> 위험도가 가장 높습니다.</div>',
    unsafe_allow_html=True
)

# =========================
# 네 개 그룹 카드
# =========================
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="card">HC<br>{probs["HC"]:.0f}%<br><span style="color:green;">건강 비교군</span></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="card">PD<br>{probs["PD"]:.0f}%<br><span style="color:blue;">파킨슨병</span></div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown(f'<div class="card">PSP<br>{probs["PSP"]:.0f}%<br><span style="color:orange;">진행성 핵상마비</span></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="card">MSA<br>{probs["MSA"]:.0f}%<br><span style="color:red; font-weight:bold;">다계통위축증 (최종)</span></div>', unsafe_allow_html=True)

# =========================
# 원형 그래프 (도넛 차트)
# =========================
st.subheader("📊 위험도 분포")

fig, ax = plt.subplots()
labels = list(probs.keys())
sizes = list(probs.values())
colors = ["#27AE60", "#2980B9", "#E67E22", "#E74C3C"]

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.0f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'width':0.4},
    textprops={'fontsize': 22, 'weight': 'bold'}   # ✅ 라벨 크기
)

# autopct (퍼센트 숫자) 크기 동일하게 맞추기
for autotext in autotexts:
    autotext.set_fontsize(22)   # ✅ 라벨과 동일 크기
    autotext.set_weight("bold")
    
ax.axis("equal")
st.pyplot(fig)

# =========================
# LLM 보고서 생성
# =========================
st.subheader("📝 AI 기반 종합 보고서")

prompt = f"""
다음은 파라솔 솔루션의 다중모달 검사 결과입니다:
{json.dumps(data, ensure_ascii=False, indent=2)}

당신은 환자 가족에게 설명하는 친절한 의사입니다.
- 네 그룹(HC, PD, PSP, MSA)의 확률을 비교해 해석하세요.
- 최종진단에 대해 쉽게 설명하세요.
- 가족이 도울 수 있는 생활 속 권장 행동을 안내하세요.
- 따뜻하고 위로가 느껴지는 톤으로 작성하세요.
"""

if st.button("📖 보고서 생성하기"):
    with st.spinner("보고서를 생성 중입니다..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        report = response.choices[0].message.content
        st.markdown(report)
        st.download_button("📥 보고서 저장하기", report, file_name="parasol_msa_report.txt")


###################
##### Chatbot

# -*- coding: utf-8 -*-
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

# ----------------------
# API 세팅
# ----------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------
# Base64 변환 함수 (아파닥 캐릭터 이미지)
# ----------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

apadoc_avatar = get_base64_image("apadoc.png")  # 같은 폴더에 apadoc.png 두기

# ----------------------
# 아파닥 챗봇 함수
# ----------------------
def apadoc_chatbot():
    # CSS 스타일 (말풍선 + 아바타)
    st.markdown("""
    <style>
    .chat-bubble-user {
        background-color: #C7F2C4;
        color: black;
        padding: 10px 14px;
        border-radius: 12px;
        margin: 6px 0;
        max-width: 70%;
        text-align: right;
        float: right;
        clear: both;
    }
    .chat-bubble-bot-wrapper {
        display: flex;
        align-items: flex-start;
        margin: 6px 0;
        clear: both;
    }
    .avatar {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        margin-right: 10px;
    }
    .chat-bubble-bot {
        background-color: #FFFFFF;
        color: black;
        padding: 10px 14px;
        border-radius: 12px;
        max-width: 70%;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title">아파닥</div>', unsafe_allow_html=True)
    st.markdown('<div class="report-box">아파닥과 대화하기</div>', unsafe_allow_html=True)

    # 세션 상태 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "안녕하세요, 아파닥입니다 무엇이 궁금하신가요?")
        ]

    # 대화 출력
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div class="chat-bubble-bot-wrapper">
                    <img src="{apadoc_avatar}" class="avatar">
                    <div class="chat-bubble-bot">{msg}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 사용자 입력창
    user_input = st.chat_input("질문을 입력하세요...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_input}]
        )
        answer = response.choices[0].message.content
        st.session_state.chat_history.append(("bot", answer))
        st.rerun()

    # FAQ 버튼
    st.subheader("📌 자주 묻는 질문")
    cols = st.columns(3)
    faq_list = ["파킨슨병이 뭐예요?", "PSP가 뭐예요?", "MSA가 뭐예요?"]
    for i, q in enumerate(faq_list):
        if cols[i].button(q):
            st.session_state.chat_history.append(("user", q))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": q}]
            )
            answer = response.choices[0].message.content
            st.session_state.chat_history.append(("bot", answer))
            st.rerun()


# ----------------------
# 버튼으로 챗봇 열기
# ----------------------
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False

if st.button("🤖 아파닥 챗봇 열기"):
    st.session_state.show_chatbot = True

# 챗봇 표시
if st.session_state.show_chatbot:
    apadoc_chatbot()
