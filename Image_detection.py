from ultralytics import YOLO

# Load model hasil training
model = YOLO('Training/weights/best.pt')  # Ganti path jika berbeda

# Path ke gambar yang ingin dideteksi
image_path = r'C:\Users\EVA-01\Pictures\kosan-sampah.jpg'
results = model(image_path)
results[0].show()
results[0].save('hasil_prediksi/')