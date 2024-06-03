import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time

# Set page title and favicon
st.set_page_config(page_title="Can Classifier", page_icon=":can:")

# Set background color and padding
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #c9d6ff, #e2e2e2);
        padding-top: 5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set title and subtitle
st.title("Can Classifier")
st.sidebar.title("Menu")
st.sidebar.write("This app classifies cans as defective or non-defective.")

# Load your Keras model
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model = load_model('model.h5')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")

model = load_model()

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
    result = 'Defective Can' if prediction[0][0] <= 0.5 else 'Non-defective Can'  # Adjust the condition as needed
    return result

# Function to check if the frame contains a can-like object
def is_valid_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return len(objects) > 0

# Function to initialize the camera with different backends
def initialize_camera(index):
    backend_candidates = [cv2.CAP_ANY, cv2.CAP_V4L, cv2.CAP_MSMF]
    for backend in backend_candidates:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            st.write(f"Camera initialized with {backend} backend.")
            return cap
    st.error("No camera detected. Please check your camera device.")
    st.stop()

# Function for login page
def login():
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        users = st.session_state["users"]
        if email in users and users[email]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = users[email]["username"]
        else:
            st.error("Invalid email or password")

# Function for registration page
def register():
    st.subheader("Register")
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

# Function for classification page
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
        camera_index = 0
        cap = initialize_camera(camera_index)

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
