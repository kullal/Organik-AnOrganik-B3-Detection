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
            st.error(f"File model tidak ditemukan di: {model_path}")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error saat memuat model: {str(e)}")
        return None

model = load_model()

# Check if model loaded successfully
if model is None:
    st.error("Gagal memuat model. Harap periksa apakah file model ada di 'Training/weights/best.pt'")
    st.stop()

# --- UI Elements ---
st.title("🌿 Deteksi Organik dan Anorganik 🧪")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    detection_mode = st.radio(
        "Pilih Mode Deteksi:",
        ["Real-time", "Upload Gambar"],
        captions = ["Deteksi objek secara langsung menggunakan kamera.", "Unggah gambar untuk dideteksi."]
    )
    st.markdown("---")
    st.markdown("**Credit : Dibuat oleh Kelompok 3**")
    st.markdown(
        """
        <div style="text-align: center;">
            <small><i>Aplikasi ini menggunakan model YOLOv8.</i></small>
        </div>
        """, unsafe_allow_html=True
    )


# Function for real-time detection
def real_time_detection(camera_index):
    st.session_state.camera_running = True
    st.session_state.stop = False
    
    col1, col2 = st.columns([3, 1]) # Create columns for video and controls

    with col1:
        stframe = st.empty()
    
    with col2:
        st.subheader("Kontrol Kamera")
        stop_button_placeholder = st.empty()
        stop_button = stop_button_placeholder.button(
            "⏹️ Stop Deteksi", key='stop_button_realtime',
            on_click=lambda: setattr(st.session_state, 'stop', True),
            use_container_width=True
        )

    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error(f"Tidak dapat mengakses kamera {camera_index}. Periksa koneksi kamera atau coba kamera lain.")
        st.session_state.camera_running = False
        return
    
    try:
        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                st.warning("Error: Tidak dapat membaca frame dari kamera. Mencoba menghubungkan kembali...")
                time.sleep(0.5) # Wait a bit before retrying
                continue # Skip this iteration
            
            frame_resized = cv2.resize(frame, (640, 480))
            results = model(frame_resized, device='cpu', verbose=False)
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            with col1: # Display in the first column
                stframe.image(annotated_frame_rgb, caption=f"Deteksi Real-time (Kamera {camera_index})", use_column_width=True)
            
            time.sleep(0.01)
    
    except Exception as e:
        st.error(f"Error dalam proses deteksi: {e}")
    finally:
        cap.release()
        st.session_state.camera_running = False
        with col2: # Clear the stop button when done
             stop_button_placeholder.empty()
        with col1:
            stframe.empty()
        st.info("Deteksi dihentikan.")

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
    
    st.image(image, caption="Gambar Asli", use_column_width=True)
    
    with st.spinner("Memproses gambar..."):
        results = model(image, device='cpu', verbose=False)
        annotated_image = results[0].plot()
    
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    st.image(annotated_image_rgb, caption="Hasil Deteksi", use_column_width=True)
    
    os.unlink(temp_file_path)

# Main functionality
if detection_mode == "Real-time":
    st.header("🎥 Deteksi Real-time")
    
    camera_options = {
        "Kamera Utama (Indeks 0)": 0,
        "Kamera Eksternal (Indeks 1)": 1
    }
    selected_camera_label = st.selectbox(
        "Pilih Sumber Kamera:",
        options=list(camera_options.keys())
    )
    camera_index = camera_options[selected_camera_label]
    
    if not st.session_state.camera_running:
        start_button = st.button("▶️ Mulai Deteksi", use_container_width=True, type="primary")
        if start_button:
            real_time_detection(camera_index)

elif detection_mode == "Upload Gambar":
    st.header("🖼️ Deteksi dari Gambar")
    uploaded_file = st.file_uploader(
        "Pilih file gambar (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)
    else:
        st.info("Silakan unggah file gambar untuk memulai deteksi.")

# Footer moved to sidebar for cleaner main page
# st.markdown("---")
# st.markdown("**Credit : Dibuat oleh Kelompok 3**")
# st.markdown("*Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi objek organik dan anorganik.*") 