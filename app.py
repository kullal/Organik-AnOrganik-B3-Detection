import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Deteksi Organik dan Anorganik",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Title
st.title("Deteksi Organik dan Anorganik")
st.markdown("### Sistem Deteksi Sampah Organik dan Anorganik Menggunakan YOLOv8")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('Training/weights/best.pt')

model = load_model()

# Create sidebar
st.sidebar.title("Pengaturan")
detection_mode = st.sidebar.radio("Mode Deteksi", ["Real-time", "Upload Gambar/Video"])

if detection_mode == "Real-time":
    # Camera selection
    camera_options = {
        "Kamera Default (0)": 0,
        "Kamera Eksternal (1)": 1
    }
    selected_camera = st.sidebar.selectbox("Pilih Kamera", list(camera_options.keys()))
    camera_id = camera_options[selected_camera]
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_button = st.button("Mulai Deteksi")
    
    with col2:
        stop_button = st.button("Hentikan Deteksi")
    
    if start_button:
        st.session_state.camera_running = True
        
    if stop_button:
        st.session_state.camera_running = False
        st.info("Deteksi dihentikan.")
    
    # Run the detection if camera is running
    if st.session_state.camera_running:
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                st.error(f"Tidak dapat membuka kamera {camera_id}. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
                st.session_state.camera_running = False
            else:
                st.success(f"Kamera {camera_id} berhasil dibuka.")
                
                # Use a "stop" button to exit the loop
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Tidak dapat membaca frame dari kamera.")
                        st.session_state.camera_running = False
                        break
                    
                    # Deteksi objek pada frame
                    results = model(frame, conf=conf_threshold, device='cpu')
                    
                    # Visualisasi hasil deteksi
                    annotated_frame = results[0].plot()
                    
                    # Convert to RGB (streamlit uses RGB)
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Show the frame
                    video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Short delay to reduce CPU usage
                    time.sleep(0.01)
                
                # Release the camera
                cap.release()
                st.info("Kamera telah dilepaskan.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.camera_running = False

elif detection_mode == "Upload Gambar/Video":
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Gambar atau Video", type=["jpg", "jpeg", "png", "mp4", "avi"])
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    
    if uploaded_file is not None:
        # Check if the file is an image or video
        file_type = uploaded_file.name.split(".")[-1].lower()
        
        if file_type in ["jpg", "jpeg", "png"]:
            # Process image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Deteksi objek pada gambar
            results = model(image_np, conf=conf_threshold, device='cpu')
            
            # Visualisasi hasil deteksi
            annotated_img = results[0].plot()
            
            # Convert to RGB if needed
            if len(annotated_img.shape) == 3 and annotated_img.shape[2] == 3:
                if file_type != "png":  # No need to convert if PNG
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Display the result
            st.image(annotated_img, caption="Hasil Deteksi", use_column_width=True)
            
        elif file_type in ["mp4", "avi"]:
            # Save the uploaded video to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}")
            temp_file.write(uploaded_file.read())
            
            # Process video
            cap = cv2.VideoCapture(temp_file.name)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create a placeholder for the video
            video_placeholder = st.empty()
            
            st.info("Memproses video... Harap tunggu.")
            
            # Process and display the video
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Deteksi objek pada frame
                results = model(frame, conf=conf_threshold, device='cpu')
                
                # Visualisasi hasil deteksi
                annotated_frame = results[0].plot()
                
                # Convert to RGB
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Show the frame
                video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
                # Control playback speed (adjust as needed)
                time.sleep(1/fps)
            
            # Release resources
            cap.release()
            st.success("Pemrosesan video selesai.")
    else:
        st.info("Silakan upload gambar atau video untuk dideteksi.")

# Add credit footer
st.markdown("---")
st.markdown("**Credit: Dibuat oleh Kelompok 3**")
st.markdown("Aplikasi deteksi sampah organik dan anorganik menggunakan YOLOv8") 