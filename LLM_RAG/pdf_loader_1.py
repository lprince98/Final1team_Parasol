# 1_pdf_loader.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

def load_and_split_pdfs(pdf_folder, chunk_size=800, chunk_overlap=100):
    docs = []
    for root, dirs, files in os.walk(pdf_folder):   # os.walk 사용
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=["\n", ".", " "]
                )
                chunks = splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    docs.append({
                        "content": chunk,
                        "source": pdf_path,   # 파일 경로 저장 (출처 확인 가능)
                        "chunk_id": i
                    })
    return docs

if __name__ == "__main__":
    pdf_folder = r"D:\workspace\Project\05_Final\paper"  # 윈도우 절대 경로
    docs = load_and_split_pdfs(pdf_folder)
    print(f"총 {len(docs)} 개 chunk 생성됨")

