# 2_build_index.py
import faiss
import numpy as np
import pickle
from openai import OpenAI
from dotenv import load_dotenv
import os
from pdf_loader_1 import load_and_split_pdfs

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(emb.data[0].embedding, dtype="float32")

if __name__ == "__main__":
    docs = load_and_split_pdfs(r"D:\workspace\Project\05_Final\paper")

    embeddings = [get_embedding(d["content"]) for d in docs]
    embeddings = np.vstack(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open("parkinson_rag_index.pkl", "wb") as f:
        pickle.dump((index, docs), f)

    print("✅ 벡터 인덱스 저장 완료: parkinson_rag_index.pkl")
