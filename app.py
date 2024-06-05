import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from threading import Thread

# Simpan kredensial pengguna di session_state (untuk demo; gunakan database nyata dalam implementasi sebenarnya)
if "users" not in st.session_state:
    st.session_state["users"] = {
        "admin@example.com": {"username": "admin", "password": "password123"}
    }

# Load your Keras model
try:
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load Haar Cascade for object detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # Placeholder path, use appropriate classifier
cascade = cv2.CascadeClassifier(cascade_path)

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
    return result

# Function to check if the frame contains a can-like object
def is_valid_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return len(objects) > 0

# Function to read frames from the camera
def read_frames():
    cap = cv2.VideoCapture(0)
    while st.session_state["logged_in"] and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if is_valid_frame(frame):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                result = predict(pil_image)
                cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            st.frame_queue.put(frame)
    cap.release()

# Fungsi untuk halaman klasifikasi
def app():
    st.title("Can Classifier")
    st.write(f"Welcome, {st.session_state['username']}!")
    st.write("This app classifies cans as defective or non-defective.")

    mode = st.radio("Choose a mode:", ('Real-Time Classification', 'Upload Picture'))

    if mode == 'Real-Time Classification':
        stframe = st.empty()
        while True:
            if not st.frame_queue.empty():
                frame = st.frame_queue.get()
                stframe.image(frame, channels="BGR")

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
    st.frame_queue = Queue()
    Thread(target=read_frames).start()
    app()
else:
    choice = st.selectbox("Login/Sign up", ["Login", "Register"])
    if choice == "Login":
        login()
    else:
        register()
