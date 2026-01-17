import streamlit as st
import numpy as np
import cv2
from PIL import Image, ExifTags
import tempfile
from database import create_table, get_user, update_user
from tensorflow import lite as tflite

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="EcoTogether",
    page_icon="♻️",
    layout="centered"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
.main { background-color: #f4fdf7; }

.card {
    background: white;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 1.2rem;
}

h1 { text-align: center; color: #2e7d32; font-weight: 800; }
h3 { color: #388e3c; }

.stButton > button {
    background: linear-gradient(135deg, #43a047, #66bb6a);
    color: white;
    border-radius: 999px;
    padding: 0.6rem 1.6rem;
    font-weight: 700;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2e7d32, #43a047);
}

img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ================== INIT ==================
create_table()

st.title("♻️ EcoTogether")
st.caption("Cùng nhau phân loại rác – tích điểm – đổi quà 🌱")
st.divider()

# ================== SIDEBAR ==================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2909/2909597.png",
        width=120
    )
    st.title("EcoTogether")
    st.caption("Hành động nhỏ – thay đổi lớn")
    st.divider()

# ================== LOGIN ==================
st.markdown("""
<div class="card" style="
    text-align:center;
    font-size:21px;
    font-weight:700;
    background: linear-gradient(90deg, #e8f5e9, #f1f8e9);
    color:#1b5e20;
">
    🌱 Chung tay bảo vệ môi trường – Vì Trái Đất xanh 🌍
</div>
""", unsafe_allow_html=True)

st.subheader("👤 Đăng nhập")
username = st.text_input("Tên người dùng")

if not username:
    st.warning("Vui lòng nhập tên để tiếp tục")
    st.stop()

st.success(f"Xin chào **{username}** 👋")
st.markdown('</div>', unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
@st.cache_resource
def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_ai():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_ai()
    st.success("🧠 AI đã sẵn sàng")
except Exception as e:
    st.error("❌ Không load được AI")
    st.code(str(e))
    st.stop()

labels = load_labels()
@st.cache_resource
def load_labels():
    with open("labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

labels = load_labels()

def predict_trash(pil_img):
    # resize đúng size model (thường 224x224)
    img = pil_img.resize((224, 224))
    img = np.asarray(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = np.argmax(output)
    confidence = float(output[idx] * 100)

    return labels[idx], confidence


# ================== EXIF CHECK ==================
def is_camera_image(pil_img):
    try:
        return pil_img._getexif() is not None
    except:
        return False

# ================== IMAGE ==================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📷 Ảnh thùng rác")

image = st.file_uploader(
    "Chụp ảnh bằng camera điện thoại rồi tải lên",
    type=["jpg", "jpeg", "png"]
)

points_image = 0

if image:
    img = Image.open(image)
    st.image(img, width=500)

    if is_camera_image(img):
        st.success("📷 Ảnh hợp lệ (từ camera)")

        with st.spinner("🧠 AI đang nhận diện rác..."):
            label, conf = predict_trash(img)

        st.markdown(f"""
        ### ♻️ Kết quả AI
        - **Loại rác:** `{label}`
        - **Độ tin cậy:** `{conf:.2f}%`
        """)

        if conf >= 60:
            points_image = 1
            st.success("✅ Ảnh hợp lệ & AI đủ tin cậy – được tính điểm")
        else:
            st.warning("⚠️ AI không đủ tin cậy – không tính điểm")

    else:
        st.error("❌ Ảnh không hợp lệ (không phải ảnh camera)")

st.markdown('</div>', unsafe_allow_html=True)


# ================== VIDEO ==================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎥 Video bỏ rác")

video = st.file_uploader(
    "Quay video bỏ rác vào thùng",
    type=["mp4", "mov"]
)

def check_motion(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    ret, prev = cap.read()
    if not ret:
        return False

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion = 0

    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion += np.sum(cv2.absdiff(prev_gray, gray))
        prev_gray = gray

    cap.release()
    return motion > 1_000_000

points_video = 0
video_valid = False

if video:
    st.video(video)
    with st.spinner("🔍 Đang kiểm tra video..."):
        video_valid = check_motion(video)

    if video_valid:
        st.success("🎥 Video hợp lệ")
        points_video = 10
    else:
        st.error("❌ Video không có chuyển động")

st.markdown('</div>', unsafe_allow_html=True)

# ================== POINTS ==================
points = 0
if points_image:
    points += 1
if video_valid:
    points += 10
if points_image and video_valid:
    points = 15

# ================== CONFIRM ==================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⭐ Xác nhận điểm")

if st.button("✅ Xác nhận & cộng điểm"):
    if points == 0:
        st.warning("Chưa đủ điều kiện nhận điểm")
    else:
        total = update_user(username, points)
        st.success(f"🎉 +{points} điểm | Tổng: {total}")
st.markdown('</div>', unsafe_allow_html=True)

# ================== TOTAL ==================
total = get_user(username)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Tổng điểm của bạn")
st.metric("⭐ Điểm", total)

st.subheader("🎁 Đổi quà")
if total >= 500:
    st.success("🎉 Bạn đủ điều kiện đổi quà 100K")
else:
    st.info(f"Còn thiếu {500 - total} điểm")
st.markdown('</div>', unsafe_allow_html=True)

