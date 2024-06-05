import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import logging
import tempfile

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

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
        return None

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image")
        return None

    cap.release()
    return frame

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
        st.write("Click the button below to capture an image from your webcam and classify it.")
        if st.button("Capture and Classify"):
            frame = capture_image()
            if frame is not None:
                st.image(frame, caption='Captured Image.', use_column_width=True)
                st.write("Classifying...")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                result = predict(pil_image)
                st.write(f"The can is **{result}**.")
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
