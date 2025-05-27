import streamlit as st
import cv2
import tempfile
import time
import os
import numpy as np
from ultralytics import YOLO
import torch

# Set page title and configuration
st.set_page_config(
    page_title="Deteksi Organik dan Anorganik",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5em;
        color: #333333;
        margin-bottom: 10px;
    }
    .footer-text {
        font-size: 0.9em;
        color: #777777;
        text-align: center;
        margin-top: 30px;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stRadio>div {
        display: flex;
        flex-direction: row;
        justify-content: center; /* Center radio buttons */
    }
    .stRadio>div>label {
        margin: 0 10px; /* Add some spacing between radio buttons */
        padding: 5px 15px;
        border: 1px solid #4CAF50;
        border-radius: 5px;
        cursor: pointer;
    }
    .stRadio>div>label:hover {
        background-color: #e6ffe6;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stop' not in st.session_state:
    st.session_state.stop = False
    
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Patch torch.load to handle the serialization issue in PyTorch 2.6+
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Apply the patch
torch.load = patched_torch_load

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    try:
        model_path = 'Training/weights/best.pt'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        return None

model = load_model()

# Page title and description
st.markdown("<p class='main-title'>🌿 Deteksi Organik dan Anorganik 🗑️</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Aplikasi ini menggunakan YOLOv8 untuk mendeteksi objek organik dan anorganik secara real-time atau melalui unggahan gambar.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan Deteksi")
    detection_mode = st.radio(
        "Pilih Mode Deteksi:",
        ["🎥 Real-time", "🖼️ Upload Gambar"],
        index=0
    )
    st.markdown("---")
    st.subheader("ℹ️ Info Model")
    if model:
        st.success("Model YOLOv8 berhasil dimuat.")
    else:
        st.error("Gagal memuat model. Harap periksa file model.")
    
    st.markdown("---")
    st.markdown("<p class='footer-text'>Dibuat oleh Kelompok 3</p>", unsafe_allow_html=True)

# Check if model loaded successfully
if model is None:
    st.error("Gagal memuat model. Aplikasi tidak dapat berjalan. Pastikan file 'Training/weights/best.pt' ada.")
    st.stop()

# Function for real-time detection
def real_time_detection(camera_index):
    st.session_state.camera_running = True
    st.session_state.stop = False
    
    col1, col2 = st.columns([3, 1]) # Column for video, column for controls

    with col2:
        st.subheader("Kontrol Kamera")
        stop_button = st.button("🛑 Hentikan Deteksi", key='stop_button_rt', on_click=lambda: setattr(st.session_state, 'stop', True))

    with col1:
        stframe_placeholder = st.empty()
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            st.error(f"Tidak dapat mengakses kamera {camera_index}. Periksa koneksi kamera atau coba kamera lain.")
            st.session_state.camera_running = False
            return
        
        try:
            while cap.isOpened() and not st.session_state.stop:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Tidak dapat membaca frame dari kamera. Menghentikan...")
                    break
                
                frame = cv2.resize(frame, (640, 480))
                results = model(frame, device='cpu')
                annotated_frame = results[0].plot()
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                stframe_placeholder.image(annotated_frame_rgb, caption=f"Deteksi Real-time (Kamera {camera_index})", use_column_width=True)
                time.sleep(0.01)
        
        except Exception as e:
            st.error(f"Error dalam proses deteksi: {e}")
        finally:
            cap.release()
            st.session_state.camera_running = False
            stframe_placeholder.empty()
            if st.session_state.stop: # Check if stop was intentional
                st.success("Deteksi real-time dihentikan.")
            st.session_state.stop = False # Reset stop state

# Function for image upload detection
def process_uploaded_image(uploaded_file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close()
    
    image = cv2.imread(temp_file_path)
    
    if image is None:
        st.error("Error: Tidak dapat membaca gambar yang diunggah. Pastikan format gambar didukung (JPG, JPEG, PNG).")
        os.unlink(temp_file_path)
        return

    st.image(image, caption="Gambar Asli", use_column_width=True, channels="BGR")

    with st.spinner("Memproses gambar..."):
        results = model(image, device='cpu')
        annotated_image = results[0].plot()
    
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    st.image(annotated_image_rgb, caption="Hasil Deteksi", use_column_width=True)
    os.unlink(temp_file_path)
    st.success("Deteksi pada gambar selesai.")

# Main functionality
if detection_mode == "🎥 Real-time":
    st.markdown("<p class='sub-header'>Deteksi Real-time menggunakan Webcam</p>", unsafe_allow_html=True)
    
    camera_options = {
        "Webcam Utama (Indeks 0)": 0,
        "Webcam Eksternal (Indeks 1)": 1
    }
    selected_camera_label = st.selectbox(
        "Pilih Sumber Kamera:",
        options=list(camera_options.keys())
    )
    camera_index = camera_options[selected_camera_label]
    
    if not st.session_state.camera_running:
        if st.button("🚀 Mulai Deteksi Real-time"):
            real_time_detection(camera_index)
    else:
        st.info("Deteksi real-time sedang berjalan. Klik 'Hentikan Deteksi' untuk berhenti.")

elif detection_mode == "🖼️ Upload Gambar":
    st.markdown("<p class='sub-header'>Deteksi dari Unggahan Gambar</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        st.markdown("---")
        process_uploaded_image(uploaded_file)

# Footer
st.markdown("---")
st.markdown("<p class='footer-text'>Aplikasi ini dikembangkan sebagai bagian dari proyek Kelompok 3 untuk mata kuliah terkait. <br> Menggunakan model YOLOv8 untuk klasifikasi objek organik dan anorganik.</p>", unsafe_allow_html=True) 