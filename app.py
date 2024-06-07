import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av

# Fungsi untuk mengekstrak fitur menggunakan HOG descriptor
def extract_hog_features(img, target_size=(300, 300)):
    # Resize gambar ke ukuran target
    img_resized = cv2.resize(img, target_size)
    
    # Ekstrak fitur menggunakan HOG descriptor
    hog = cv2.HOGDescriptor(_winSize=(300, 300),
                            _blockSize=(100, 100),
                            _blockStride=(50, 50),
                            _cellSize=(50, 50),
                            _nbins=9)
    h = hog.compute(img_resized)
    
    return h.flatten() if h is not None else None

# Load model SVM dari file XML
svm = cv2.ml.SVM_load('svm_defect_model.xml')

# Fungsi untuk memprediksi menggunakan model SVM
def predict(image):
    # Konversi gambar ke grayscale
    img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Ekstrak fitur HOG dari gambar
    features = extract_hog_features(img_gray)
    if features is None:
        raise ValueError(f"Gagal mengekstrak fitur dari gambar.")
    
    # Lakukan prediksi
    features = np.array([features], dtype=np.float32)
    _, result = svm.predict(features)
    return int(result[0][0])

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        try:
            label = predict(img)
            if label == 0:
                text = "Kaleng Tidak Cacat"
                color = (0, 255, 0)  # Green for non-defective
            else:
                text = "Kaleng Cacat"
                color = (0, 0, 255)  # Red for defective
        except Exception as e:
            text = f"Error: {e}"
            color = (255, 0, 0)  # Blue for error
        
        # Tambahkan teks ke gambar
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("Deteksi Kaleng Cacat atau Tidak Cacat secara Real-Time")
st.write("Gunakan webcam untuk melakukan prediksi real-time.")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Untuk menjalankan aplikasi Streamlit, gunakan perintah berikut di terminal
# streamlit run app.py
