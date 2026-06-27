import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Debug API", layout="wide")
st.title("🔍 Trạm Chẩn Đoán Gemini API")

try:
    # 1. Kiểm tra cấu hình Secrets
    if "GEMINI_API_KEY" not in st.secrets:
        st.error(
            "❌ Không tìm thấy biến 'GEMINI_API_KEY'. Bạn hãy kiểm tra lại mục Settings -> Secrets trên Streamlit Cloud."
        )
        st.stop()

    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    st.success(
        f"✅ Đã đọc được API Key. Bắt đầu bằng: `{api_key[:15]}...` (Độ dài: {len(api_key)} ký tự)"
    )

    # 2. Quét danh sách Model mà Google cấp cho Key này
    st.subheader("📦 Các Model mà Google cho phép bạn sử dụng:")
    with st.spinner("Đang kết nối với Google..."):
        models = genai.list_models()
        allowed_models = [m.name for m in models]
        st.write(allowed_models)

    # 3. Test trực tiếp
    st.subheader("🧪 Chạy thử tính năng Embedding")
    test_model = "models/text-embedding-004"

    if test_model in allowed_models:
        res = genai.embed_content(model=test_model, content="Kiểm tra hệ thống API.")
        st.success(f"🎉 TUYỆT VỜI! Đã nhúng thành công vector bằng model {test_model}.")
    else:
        st.error(f"❌ API Key của bạn bị CẤM sử dụng model '{test_model}'.")
        st.warning(
            "👉 Giải pháp dứt điểm: Bạn bắt buộc phải dùng một tài khoản Gmail khác đăng nhập vào Google AI Studio để tạo lại API Key mới."
        )

except Exception as e:
    st.error(f"🚨 Lỗi từ phía Google: {e}")
