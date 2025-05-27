import streamlit as st
import cv2
import tempfile
import time
import os
import numpy as np
from ultralytics import YOLO

# Set page title and configuration
st.set_page_config(
    page_title="Deteksi Organik dan Anorganik",
    layout="wide"
)

# Initialize session state
if 'stop' not in st.session_state:
    st.session_state.stop = False
    
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('Training/weights/best.pt')

model = load_model()

# Page title and description
st.title("Deteksi Organik dan Anorganik")
st.markdown("Aplikasi ini menggunakan YOLOv8 untuk mendeteksi objek organik dan anorganik.")

# Sidebar
st.sidebar.title("Pengaturan")
detection_mode = st.sidebar.radio("Pilih Mode Deteksi:", ["Real-time", "Upload Gambar"])

# Function for real-time detection
def real_time_detection(camera_index):
    st.session_state.camera_running = True
    st.session_state.stop = False
    
    # Create a stop button
    stop_button_placeholder = st.empty()
    stop_button = stop_button_placeholder.button(
        "Stop", key='stop_button',
        on_click=lambda: setattr(st.session_state, 'stop', True)
    )
    
    stframe = st.empty()
    cap = cv2.VideoCapture(camera_index)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        st.error(f"Tidak dapat mengakses kamera {camera_index}. Periksa koneksi kamera atau coba kamera lain.")
        st.session_state.camera_running = False
        return
    
    try:
        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Tidak dapat membaca frame dari kamera.")
                break
            
            # Resize for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Perform detection
            results = model(frame, device='cpu')
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Convert to RGB for streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            stframe.image(annotated_frame_rgb, caption=f"Deteksi Real-time (Kamera {camera_index})", use_column_width=True)
            
            # Add a small sleep to reduce CPU usage
            time.sleep(0.01)
    
    except Exception as e:
        st.error(f"Error dalam proses deteksi: {e}")
    finally:
        cap.release()
        st.session_state.camera_running = False
        stop_button_placeholder.empty()
        stframe.empty()
        st.success("Deteksi dihentikan")

# Function for image upload detection
def process_uploaded_image(uploaded_file):
    # Create a temporary file to save the uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Read the image
    image = cv2.imread(temp_file_path)
    
    if image is None:
        st.error("Error: Tidak dapat membaca gambar yang diunggah. Pastikan format gambar didukung.")
        os.unlink(temp_file_path)
        return
    
    # Perform detection
    results = model(image, device='cpu')
    
    # Draw results on image
    annotated_image = results[0].plot()
    
    # Convert to RGB for streamlit
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Display the result
    st.image(annotated_image_rgb, caption="Hasil Deteksi", use_column_width=True)
    
    # Clean up the temporary file
    os.unlink(temp_file_path)

# Main functionality
if detection_mode == "Real-time":
    st.header("Deteksi Real-time")
    
    camera_options = {
        "Webcam Utama (0)": 0,
        "Webcam Eksternal (1)": 1
    }
    selected_camera = st.selectbox(
        "Pilih Kamera:",
        options=list(camera_options.keys())
    )
    camera_index = camera_options[selected_camera]
    
    # Start/Stop button logic
    if not st.session_state.camera_running:
        start_button = st.button("Mulai Deteksi")
        if start_button:
            real_time_detection(camera_index)

else:  # Upload Image mode
    st.header("Deteksi dari Gambar")
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        process_uploaded_image(uploaded_file)

# Footer
st.markdown("---")
st.markdown("**Credit : Dibuat oleh Kelompok 3**")
st.markdown("*Aplikasi ini menggunakan model YOLOv8 untuk mendeteksi objek organik dan anorganik.*") 