import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import logging
import asyncio

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Simpan kredensial pengguna di session_state (untuk demo; gunakan database nyata dalam implementasi sebenarnya)
if "users" not in st.session_state:
    st.session_state["users"] = {
        "admin@example.com": {"username": "admin", "password": "password123"}
    }

# Load your Keras model
try:
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    logging.error(f"Error loading model: {e}")

# Load Haar Cascade for object detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Placeholder path, use appropriate classifier
cascade = cv2.CascadeClassifier(cascade_path)
logging.info("Cascade classifier loaded successfully")

# Function to preprocess the image/frame and make predictions
def preprocess_image(image):
    image = image.resize((300, 300))  # Adjust target_size as needed
    image_array = np.array(image) / 255.0  # Normalize if needed
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    result = 'Kaleng Cacat' if prediction[0][0] <= 0.5 else 'Kaleng Tidak Cacat'  # Adjust the condition as needed
    logging.info(f"Prediction made: {result}")
    return result

# Function to check if the frame contains a can-like object
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
        logging.info("Frame received for transformation")

        # Increment frame counter
        self.frame_counter += 1
        logging.info(f"Processing frame {self.frame_counter}")

        # Check if the frame contains a can-like object
        if is_valid_frame(img):
            logging.info("Valid frame detected")
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            result = predict(pil_image)
            cv2.putText(img, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            logging.info(f"Classification result: {result}")

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
            logging.info(f"User {email} logged in")
        else:
            st.error("Invalid email or password")
            logging.warning(f"Failed login attempt for email: {email}")

# Fungsi untuk halaman register
def register():
    st.title("Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    if st.button("Register"):
        users = st.session_state["users"]
        if email in users:
            st.error("Email already registered")
            logging.warning(f"Registration attempt with already registered email: {email}")
        else:
            users[email] = {"username": username, "password": password}
            st.session_state["users"] = users
            st.success("Registration successful. Please log in.")
            logging.info(f"New user registered with email: {email}")

# Fungsi untuk halaman klasifikasi
def app():
    st.title("Can Classifier")
    st.write(f"Welcome, {st.session_state['username']}!")
    st.write("This app classifies cans as defective or non-defective.")

    mode = st.radio("Choose a mode:", ('Real-Time Classification', 'Upload Picture'))

    if mode == 'Real-Time Classification':
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
    elif mode == 'Upload Picture':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            result = predict(image)

            st.write(f"The can is **{result}**.")

# Main loop
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    app()
else:
    choice = st.selectbox("Login/Sign up", ["Login", "Register"])
    if choice == "Login":
        login()
    else:
        register()
