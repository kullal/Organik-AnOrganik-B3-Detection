# Aplikasi Deteksi Sampah Organik dan Anorganik

Aplikasi ini dibuat untuk mendeteksi sampah organik dan anorganik menggunakan model YOLOv8 yang telah dilatih sebelumnya.

## Fitur

- **Deteksi Real-time**: Deteksi sampah organik dan anorganik melalui webcam atau kamera eksternal secara langsung
- **Switch Kamera**: Kemampuan beralih antara kamera default (0) dan kamera eksternal (1)
- **Upload Gambar/Video**: Analisis gambar atau video yang diunggah
- **Pengaturan Confidence**: Mengatur ambang batas kepercayaan deteksi

## Cara Menjalankan Aplikasi

1. Pastikan Python 3.8+ telah terinstal
2. Instal dependensi yang diperlukan:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```
   streamlit run app.py
   ```
4. Aplikasi akan terbuka di browser secara otomatis

## Penggunaan

### Mode Real-time

1. Pilih "Real-time" pada sidebar
2. Pilih kamera yang ingin digunakan (Default atau Eksternal)
3. Atur confidence threshold sesuai kebutuhan
4. Klik "Mulai Deteksi" untuk memulai
5. Klik "Hentikan Deteksi" untuk menghentikan proses

### Mode Upload Gambar/Video

1. Pilih "Upload Gambar/Video" pada sidebar
2. Upload file gambar (jpg, jpeg, png) atau video (mp4, avi)
3. Atur confidence threshold sesuai kebutuhan
4. Hasil deteksi akan ditampilkan secara otomatis

## Struktur Proyek

- `app.py` - Kode utama aplikasi Streamlit
- `requirements.txt` - Daftar dependensi yang diperlukan
- `Training/weights/best.pt` - Model YOLOv8 terlatih

## Persyaratan Sistem

- Python 3.8 atau lebih tinggi
- Webcam/kamera untuk mode real-time
- RAM minimal 4GB
- Ruang disk minimal 1GB

## Dibuat Oleh

Kelompok 3 