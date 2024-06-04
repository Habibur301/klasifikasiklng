import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import tensorflow as tf

# Simpan kredensial pengguna di session_state (untuk demo; gunakan database nyata dalam implementasi sebenarnya)
if "users" not in st.session_state:
    st.session_state["users"] = {
        "admin@example.com": {"username": "admin", "password": "password123"}
    }

# Load your Keras model
try:
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

# Function to initialize the camera with different backends
def initialize_camera(index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
    return None

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
        else:
            st.error("Invalid email or password")

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
        else:
            users[email] = {"username": username, "password": password}
            st.session_state["users"] = users
            st.success("Registration successful. Please log in.")

# Fungsi untuk halaman klasifikasi
def app():
    st.title("Can Classifier")
    st.write(f"Welcome, {st.session_state['username']}!")
    st.write("This app classifies cans as defective or non-defective.")

    mode = st.radio("Choose a mode:", ('Real-Time Classification', 'Upload Picture'))

    if mode == 'Real-Time Classification':
        run = st.checkbox('Run')
        FRAME_WINDOW = st.image([])
        RESULT_WINDOW = st.empty()

        # Initialize camera
        cap = initialize_camera(0)
        if not cap:
            st.error("No camera detected at index 0 or 1. Please check your camera device.")
            st.stop()

        while run:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture frame from camera. Please check your camera device.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if is_valid_frame(frame):
                pil_image = Image.fromarray(frame_rgb)

                result = predict(pil_image)

                # Display the result image and the classification result
                col1, col2 = st.columns(2)
                with col1:
                    FRAME_WINDOW.image(frame_rgb, caption="Captured Image")
                with col2:
                    RESULT_WINDOW.markdown(f"### Result Image\n\n**{result}**")
            else:
                # Clear the result window if the frame is not valid
                RESULT_WINDOW.empty()

        cap.release()
        cv2.destroyAllWindows()

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
