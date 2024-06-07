import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import logging

# Inisialisasi logging
logging.basicConfig(level=logging.INFO)

# Simpan kredensial pengguna di session_state (untuk demo; gunakan database nyata dalam implementasi sebenarnya)
if "users" not in st.session_state:
    st.session_state["users"] = {
        "admin@example.com": {"username": "admin", "password": "password123"}
    }

# Load model Keras
try:
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Model berhasil dimuat")
except Exception as e:
    st.error(f"Error loading model: {e}")
    logging.error(f"Error loading model: {e}")

# Load Haar Cascade untuk deteksi objek
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Path placeholder, gunakan classifier yang sesuai
cascade = cv2.CascadeClassifier(cascade_path)
logging.info("Cascade classifier berhasil dimuat")

# Fungsi untuk pra-pemrosesan gambar/frame dan membuat prediksi
def preprocess_image(image):
    try:
        image = image.resize((300, 300))  # Sesuaikan target_size jika perlu
        image_array = np.array(image) / 255.0  # Normalisasi jika perlu
        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]  # Konversi RGBA ke RGB jika perlu
        image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
        logging.info(f"Image preprocessed: {image_array.shape}")
        return image_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        st.error(f"Error preprocessing image: {e}")

def predict(image):
    try:
        processed_image = preprocess_image(image)
        if processed_image is not None:
            prediction = model.predict(processed_image)
            result = 'Kaleng Cacat' if prediction[0][0] <= 0.5 else 'Kaleng Tidak Cacat'  # Sesuaikan kondisinya jika perlu
            logging.info(f"Prediction made: {result}")
            return result
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error(f"Error during prediction: {e}")

# Fungsi untuk memeriksa apakah frame mengandung objek seperti kaleng
def is_valid_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    logging.info(f"Objects detected: {len(objects)}")
    return len(objects) > 0

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        logging.info("Frame diterima untuk transformasi")

        # Tambahkan penghitung frame
        self.frame_counter += 1
        logging.info(f"Memproses frame {self.frame_counter}")

        # Periksa apakah frame mengandung objek seperti kaleng
        if is_valid_frame(img):
            logging.info("Frame valid terdeteksi")
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            result = predict(pil_image)
            cv2.putText(img, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            logging.info(f"Hasil klasifikasi: {result}")

        return img

# Fungsi untuk halaman login
def login():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        users = st.session_state["users"]
        if email in users and users[email]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = users[email]["username"]
            logging.info(f"User {email} berhasil login")
        else:
            st.error("Email atau password salah")
            logging.warning(f"Gagal login untuk email: {email}")

# Fungsi untuk halaman register
def register():
    st.title("Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    if st.button("Register"):
        users = st.session_state["users"]
        if email in users:
            st.error("Email sudah terdaftar")
            logging.warning(f"Pendaftaran dengan email yang sudah terdaftar: {email}")
        else:
            users[email] = {"username": username, "password": password}
            st.session_state["users"] = users
            st.success("Pendaftaran berhasil. Silakan login.")
            logging.info(f"Pengguna baru terdaftar dengan email: {email}")

# Fungsi untuk halaman klasifikasi
def app():
    st.title("Can Classifier")
    st.write(f"Selamat datang, {st.session_state['username']}!")
    st.write("Aplikasi ini mengklasifikasikan kaleng sebagai cacat atau tidak cacat.")

    mode = st.radio("Pilih mode:", ('Klasifikasi Real-Time', 'Unggah Gambar'))

    if mode == 'Klasifikasi Real-Time':
        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
        )
    elif mode == 'Unggah Gambar':
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
            st.write("")
            st.write("Mengklasifikasikan...")

            result = predict(image)

            st.write(f"Kaleng tersebut **{result}**.")

# Main loop
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    app()
else:
    choice = st.selectbox("Login/Daftar", ["Login", "Register"])
    if choice == "Login":
        login()
    else:
        register()
