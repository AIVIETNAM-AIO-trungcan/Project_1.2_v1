# Phần 1: Import cấu hình
import streamlit as st
import tempfile, os, time
import pypdf
import chromadb
import google.generativeai as genai

# Cấu hình API Key của Google Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời: """

# Phần 2: Khởi tạo session state
for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
    st.session_state.setdefault(k, v)


# Phần 3: Các phần xử lý (core functions)
def embed(texts):
    """Chuyển danh sách chuỗi text thành danh sách vector bằng Gemini"""
    result = genai.embed_content(
        model="models/text-embedding-004", content=texts, task_type="retrieval_document"
    )
    return result["embedding"]


def chunk_text(text, size=1000, overlap=200):
    """Cắt text thành các đoạn nhỏ có gối đầu"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def process_pdf(uploaded_file):
    """Đọc PDF, cắt nhỏ, tạo embedding và lưu vào ChromaDB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(path).pages)
    os.unlink(path)

    chunks = chunk_text(text)

    #  Ngăn chặn lỗi văng app khi PDF là file scan/ảnh rỗng chữ
    if not chunks:
        st.error(
            "❌ Không tìm thấy văn bản! Có thể đây là file scan/ảnh. Vui lòng upload PDF chứa chữ thuần."
        )
        st.stop()

    client = chromadb.Client()
    collection = client.get_or_create_collection(f"rag_{int(time.time())}")

    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
    )
    return collection, len(chunks)


def rag(question, collection, k=4):
    """Hàm RAG chính: Tìm context rồi hỏi LLM Gemini."""
    # 1. Nhúng câu hỏi của người dùng thành vector
    query_emb = genai.embed_content(
        model="models/text-embedding-004", content=question, task_type="retrieval_query"
    )["embedding"]

    # 2. Tìm kiếm trong ChromaDB
    res = collection.query(query_embeddings=[query_emb], n_results=k)
    context = "\n\n".join(res["documents"][0])

    # 3. Gửi Context và Câu hỏi cho Gemini 1.5 Flash trả lời
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt_text = PROMPT.format(context=context, question=question)
    response = model.generate_content(prompt_text)

    return response.text


# Phần 4: Giao diện UI
st.set_page_config(
    page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded"
)
st.title("PDF RAG Assistant: Cloud Version ☁️")

with st.sidebar:
    st.subheader("Upload tài liệu")
    f = st.file_uploader("Chọn file PDF", type="pdf")
    if f and st.button("Xử lý File PDF", use_container_width=True):
        with st.spinner("Đang xử lý (Gemini API)..."):
            st.session_state.collection, n = process_pdf(f)
            st.session_state.pdf_name = f.name
            st.session_state.chat_history = []
        st.success(f"{n} chunks")
    st.info(
        f"📄 {st.session_state.pdf_name}"
        if st.session_state.pdf_name
        else "Chưa có tài liệu"
    )
    if st.button("Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if st.session_state.collection is None:
    st.info("Upload và xử lý PDF trước khi chat")
    st.chat_input("Nhập câu hỏi ...", disabled=True)
else:
    q = st.chat_input("Nhập câu hỏi của bạn ...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                ans = rag(q, st.session_state.collection)
                st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
