# -*- coding: utf-8 -*-
import streamlit as st
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd

# =========================
# .env 로드
# =========================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ OPENAI_API_KEY를 불러오지 못했습니다. .env 파일을 확인하세요.")
client = OpenAI(api_key=api_key)

# =========================
# Streamlit 페이지 설정
# =========================
st.set_page_config(layout="centered", page_title="파라솔 다중모달 검사 보고서")

# ✅ 카드 UI 스타일 (모바일 최적화)
st.markdown("""
<style>
.report-card {
    background-color: #ffffff;
    padding: 20px;
    margin: 14px auto;
    border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    font-size: 18px;
    line-height: 1.6;
    max-width: 420px;
}
.report-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 10px;
}
button, .stDownloadButton button {
    width: 100%;
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("📱 파라솔 다중모달 검사 보고서")

# =========================
# 샘플 JSON (MSA 결과 예시)
# =========================
data = {
    "individual_analyses": {
        "eye_tracking": {
            "summary": {
                "severity_score": 8.3,
                "interpretation": "수직 안구 운동 속도가 현저히 느려지고 불안정성이 관찰됨"
            }
        },
        "finger_tapping": {
            "summary": {
                "severity_score": 7.9,
                "interpretation": "손가락 부딪치기 속도와 진폭이 감소, 좌우 비대칭성 관찰됨"
            }
        },
        "voice_analysis": {
            "summary": {
                "severity_score": 8.4,
                "interpretation": "목소리가 점차 작아지고 발음이 불분명해짐"
            }
        }
    },
    "comprehensive_results": {
        "group_probabilities": {
            "HC (정상)": 0.05,
            "PD (파킨슨병)": 0.12,
            "PSP (진행성핵상마비)": 0.11,
            "MSA (다계통위축증)": 0.72
        },
        "final_diagnosis": "MSA (다계통위축증)",
        "summary": {
            "executive_summary": "검사 결과, MSA의 특징적 양상이 두드러지게 나타났습니다.",
            "recommendations": "전문의 상담을 통해 추가적인 영상검사 및 치료 계획 수립이 필요합니다.",
            "next_steps": "가족의 일상 지원 준비, 정기적 신경과 진료, 운동 및 재활 병행"
        }
    }
}

# -------------------------------
# 검사별 카드 표시
# -------------------------------
st.subheader("📊 검사별 결과")

analysis_map = {
    "eye_tracking": "👁️ 수직 안구 운동",
    "finger_tapping": "✋ 손가락 부딪치기",
    "voice_analysis": "🎤 소리 내기"
}

for key, label in analysis_map.items():
    summary = data["individual_analyses"][key]["summary"]
    st.markdown(f"""
    <div class="report-card">
        <div class="report-title">{label}</div>
        심각도 점수: {summary["severity_score"]:.1f} / 10<br>
        📝 해석: {summary["interpretation"]}
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# 그룹별 비교표
# -------------------------------
st.subheader("📊 그룹별 확률 비교")

probs = data["comprehensive_results"]["group_probabilities"]
df_probs = pd.DataFrame(list(probs.items()), columns=["그룹", "확률"])
df_probs["확률(%)"] = df_probs["확률"] * 100

st.table(df_probs[["그룹", "확률(%)"]])

# -------------------------------
# 종합 판단
# -------------------------------
st.subheader("🏥 종합 판단")
summary = data["comprehensive_results"]["summary"]

st.markdown(f"""
<div class="report-card">
    <div class="report-title">최종 진단</div>
    {data["comprehensive_results"]["final_diagnosis"]}
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="report-card">
    <div class="report-title">🧾 요약</div>
    {summary.get("executive_summary", "")}
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="report-card">
    <div class="report-title">📌 권장 사항</div>
    {summary.get("recommendations", "")}
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="report-card">
    <div class="report-title">➡️ 다음 단계</div>
    {summary.get("next_steps", "")}
</div>
""", unsafe_allow_html=True)

# -------------------------------
# LLM 보고서 생성
# -------------------------------
st.subheader("📝 AI 기반 종합 보고서")

prompt = f"""
다음은 파라솔 솔루션의 다중모달 검사 결과입니다:
{json.dumps(data, ensure_ascii=False, indent=2)}

당신은 환자 가족에게 설명하는 건강 상담가입니다.
- 결과를 요약하고,
- 네 그룹(HC, PD, PSP, MSA) 비교표를 해석해 주세요.
- 최종 진단(MSA)을 이해하기 쉽게 설명해 주세요.
- 가족이 도울 수 있는 생활 속 권장 행동을 안내해 주세요.
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
