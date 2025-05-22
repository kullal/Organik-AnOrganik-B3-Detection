# Organik dan Anorganik Detection

## 📋 Deskripsi Project
Project ini menggunakan YOLOv8 untuk mendeteksi dan mengklasifikasikan sampah ke dalam tiga kategori:
- Organik
- Anorganik
- B3 (Bahan Berbahaya dan Beracun)

Sistem ini dapat digunakan untuk:
- Deteksi sampah dari gambar
- Deteksi sampah secara real-time melalui webcam
- Meningkatkan efisiensi pengelolaan sampah

## ✨ Fitur
- Deteksi multi-kelas (Organik, Anorganik, B3)
- Real-time detection menggunakan webcam
- Support untuk analisis gambar dan video
- Menggunakan YOLOv8, state-of-the-art untuk object detection

## 🛠️ Teknologi
- Python 3.x
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Torchvision

## 🚀 Cara Menggunakan

### Persyaratan
```bash
pip install ultralytics opencv-python
```

### Deteksi Gambar
```python
from ultralytics import YOLO

model = YOLO('Training/weights/best.pt')
image_path = 'path/to/image.jpg'

results = model(image_path)
results[0].show()  # Tampilkan hasil
results[0].save('hasil_prediksi/')  # Simpan hasil
```

### Deteksi Real-time
```python
from ultralytics import YOLO
import cv2
import torch

# Load model
model = YOLO('Training/weights/best.pt')

# Buka webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek (gunakan CPU jika ada error CUDA)
    results = model(frame, device='cpu')
    
    # Visualisasi hasil
    annotated_frame = results[0].plot()
    
    # Tampilkan
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📂 Struktur Project
```
project/
├── train/          # Training dataset
│   ├── images/
│   └── labels/
│
├── valid/          # Validation dataset
│   ├── images/
│   └── labels/
│
├── test/           # Test dataset
│   ├── images/
│   └── labels/
│
├── Training/       # Hasil training
│   └── weights/
│        └── best.pt
│
├── Image_detection.py  # Script deteksi gambar
│
├── realtime.py     # Script deteksi real-time
│
├── data.yaml       # Konfigurasi dataset
│
└── README.md
```

## 🔄 Training Model
Model dilatih menggunakan GPU di Google Colab dengan dataset yang telah diannotasi. Parameter training:
- Model dasar: YOLOv8n
- Epochs: 10
- Image size: 640
- Batch size: 16

## 📊 Performa
- mAP@0.5: [nilai]
- Precision: [nilai]
- Recall: [nilai]
- FPS pada CPU: ~20-30 FPS

## 📝 To-Do
- [ ] Menambahkan deteksi untuk lebih banyak sub-kategori
- [ ] Mengoptimalkan model untuk perangkat dengan performa rendah
- [ ] Mengimplementasikan dalam aplikasi mobile

## 👥 Kontribusi
Kontribusi dan saran sangat diterima. Silakan buat issue atau pull request untuk berkontribusi.

---
 
