# Phần 1: Import cấu hình
import streamlit as st
import tempfile, os, time
import pypdf
import chromadb
import ollama

LLM_MODEL = "vicuna:7b-v1.5-q5_1"
EMBED_MODEL = "bge-m3"

PROMPT = """ Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngũ cảnh không có thông tin, hãy nói bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời: """

# Phần 2: Khởi tạo session state
for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
    st.session_state.setdefault(k, v)


# Phần 3: Các phần xử lý (core functions)
def embed(texts):  # từ str thanh array
    """
    Chuyển danh sách chuổi text thành danh sách vector
    """
    return ollama.embed(model="bge-m3", input=texts)["embeddings"]


def chunk_text(text, size=1000, overlap=200):
    """
    Cắt text thành các đoạn nhỏ  có độ dài tối đa là 'size' ký tự,
    vời 'overlap' ký tự trùng lặp giữa 2 đoạn lệnh liên tiếp
    """
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
    """Đọc PDF, cắt nhỏ, tạp embedding và lưu vào ChrmaDB."""
    # Lưu file upload thành file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    # Đọc nội dụng PDF
    text = "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(path).pages)
    os.unlink(path)  # Xóa file tạm

    # Cắt nhỏ và lưu vào ChromaDB
    chunks = chunk_text(text)
    client = chromadb.Client()
    collection = client.get_or_create_collection(f"rag_{int(time.time())}")

    # Thêm tất cả chunks vào database
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
    )
    return collection, len(chunks)


def rag(question, collection, k=4):
    """Hàm RAG chính: Tìm context rồi hỏi LLM."""
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    context = "\n\n".join(res["documents"][0])
    resp = ollama.chat(
        model="vicuna:7b-v1.5-q5_1",
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(context=context, question=question),
            }
        ],
        options={"temperature": 0},
    )
    return resp["message"]["content"]


# Phần 4: Giao diện  UI
# Cấu hình trang
st.set_page_config(
    page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded"
)
st.title("PDF RAG Assistant: Native")

# Sidebar:  upload PDF và nút điều khiển
with st.sidebar:
    st.subheader("Upload tài liệu")
    f = st.file_uploader("Chọn file PDF", type="pdf")
    if f and st.button("Xử lý File PDF", use_container_width=True):
        with st.spinner("Đang xử lý ..."):
            st.session_state.collection, n = process_pdf(f)
            st.session_state.pdf_name = f.name
            st.session_state.chat_history = []
        st.success(f"{n} chunks")
    st.info(
        f" {st.session_state.pdf_name}"
        if st.session_state.pdf_name
        else "Chưa có tài liệu"
    )
    if st.button("Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []

# Hiển thị lịch sử chat
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Ô nhập câu hỏi
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
