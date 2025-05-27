from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO('Training/weights/best.pt')  # Ganti path jika berbeda

# Buka webcam (0 = webcam default)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek pada frame
    results = model(frame, device='cpu')

    # Visualisasi hasil deteksi pada frame
    annotated_frame = results[0].plot()  # hasil plot bounding box

    # Tampilkan frame hasil deteksi
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()