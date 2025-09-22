# 3_rag_utils.py
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("parkinson_rag_index.pkl", "rb") as f:
    index, documents = pickle.load(f)

def embed_query(text):
    emb = client.embeddings.create(model="text-embedding-3-large", input=text)
    return np.array(emb.data[0].embedding, dtype="float32").reshape(1, -1)

def retrieve_and_answer(query, k=3):
    q_emb = embed_query(query)
    D, I = index.search(q_emb, k)
    retrieved = [documents[i]["content"] for i in I[0]]
    context = "\n\n".join(retrieved)

    prompt = f"""
    당신은 환자와 가족을 돕는 건강 상담가입니다.
    아래 질문에 대해 따뜻하고 이해하기 쉽게 설명하세요.
    질문: {query}

    참고자료:
    {context}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content
