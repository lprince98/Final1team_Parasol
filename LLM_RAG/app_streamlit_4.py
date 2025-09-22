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
# Base64 변환 함수
# ----------------------
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

# ✅ 아바닥 거북이 이미지
apadoc_avatar = get_base64_image("apadoc.png")

# ----------------------
# CSS 스타일
# ----------------------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    color: white;
    background-color: #2C3E91;
    padding: 10px;
    border-radius: 8px;
}
.report-box {
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    color: black;
    background-color: #FFD6C9;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
}
.chat-bubble-user {
    background-color: #C7F2C4;  /* 연두색 */
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
    width: 100px;   /* ✅ 크기 키움 */
    height: 100px;
    border-radius: 50%;
    margin-right: 10px;
}
.chat-bubble-bot {
    background-color: #FFFFFF;  /* 흰색 */
    color: black;
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 70%;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# ----------------------
# 타이틀
# ----------------------
st.markdown('<div class="title">아파닥</div>', unsafe_allow_html=True)
st.markdown('<div class="report-box">아파닥과 대화하기</div>', unsafe_allow_html=True)

# ----------------------
# 세션 상태 초기화
# ----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("bot", "안녕하세요, 아파닥입니다 무엇이 궁금하신가요?")
    ]

# ----------------------
# 대화 출력
# ----------------------
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

# ----------------------
# 사용자 입력창
# ----------------------
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

# ----------------------
# FAQ 버튼
# ----------------------
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
