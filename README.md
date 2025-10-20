# 🌾 Wereng Classification API

Backend API untuk klasifikasi hama wereng pada tanaman padi menggunakan Deep Learning (CNN).

## 📋 Fitur

- ✅ Upload gambar untuk klasifikasi hama wereng
- ✅ Prediksi jenis wereng (Wereng Coklat, Hijau, Punggung Putih, atau Bukan Wereng)
- ✅ Batch classification (multiple images)
- ✅ Model info dan metadata
- ✅ Prediction history logging
- ✅ REST API dengan dokumentasi Swagger UI
- ✅ Support JPG, JPEG, PNG

## 🗂️ Struktur Proyek

```
wereng-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py                      # Entry point FastAPI
│   ├── models/
│   │   └── wereng_classifier.h5     # Model CNN (belum ada, buat dulu)
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── classify.py              # Endpoint klasifikasi
│   │   └── info.py                  # Endpoint info model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Image preprocessing
│   │   └── helper.py                # Helper functions
│   └── static/
│       └── uploads/                 # Temporary upload folder
│
├── logs/
│   └── prediction_logs.txt          # Log prediksi
│
├── requirements.txt
├── README.md
└── model_training.ipynb             # Notebook training model
```

## 🚀 Instalasi dan Setup

### 1. Clone atau Download Project

```bash
# Buat folder project
mkdir wereng-api
cd wereng-api
```

### 2. Buat Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Catatan:** Jika Anda tidak memerlukan GPU, gunakan `tensorflow-cpu` untuk instalasi lebih cepat:

```bash
pip uninstall tensorflow
pip install tensorflow-cpu==2.15.0
```

### 4. Buat Struktur Folder

```bash
# Windows
mkdir app\models app\routes app\utils app\static\uploads logs

# Linux/Mac
mkdir -p app/models app/routes app/utils app/static/uploads logs
```

### 5. Buat File **init**.py

```bash
# Windows
type nul > app\__init__.py
type nul > app\routes\__init__.py
type nul > app\utils\__init__.py

# Linux/Mac
touch app/__init__.py
touch app/routes/__init__.py
touch app/utils/__init__.py
```

## ▶️ Menjalankan Server

```bash
# Development mode dengan auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Atau jalankan langsung
python -m app.main
```

Server akan berjalan di: **http://localhost:8000**

## 📚 API Documentation

Setelah server berjalan, buka dokumentasi interaktif:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### 1. Root Endpoint

```
GET /
```

Cek status API

**Response:**

```json
{
  "status": "active",
  "message": "Wereng Classification API is running",
  "timestamp": "2025-10-20T21:30:00"
}
```

### 2. Classify Image

```
POST /api/classify
```

Upload gambar untuk klasifikasi

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file: JPG, PNG)

**Response:**

```json
{
  "success": true,
  "prediction": {
    "label": "Wereng Coklat (Brown Planthopper)",
    "confidence": 0.94,
    "class_id": 0
  },
  "filename": "wereng_sample.jpg",
  "timestamp": "2025-10-20T21:30:00"
}
```

### 3. Batch Classification

```
POST /api/classify/batch
```

Upload multiple images (max 10)

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `files` (multiple image files)

### 4. Model Info

```
GET /api/model/info
```

Informasi model (akurasi, arsitektur, dll)

**Response:**

```json
{
  "success": true,
  "model_loaded": true,
  "model_info": {
    "model_name": "Wereng Classifier",
    "architecture": "MobileNetV2",
    "training_accuracy": 0.95,
    "validation_accuracy": 0.92,
    "total_classes": 4,
    "classes": [...]
  }
}
```

### 5. Prediction History

```
GET /api/history?limit=50
```

Riwayat prediksi

### 6. Clear History

```
DELETE /api/history
```

Hapus riwayat prediksi

## 🧪 Testing dengan cURL

```bash
# Test root endpoint
curl http://localhost:8000/

# Test classification
curl -X POST http://localhost:8000/api/classify \
  -F "file=@path/to/your/image.jpg"

# Test model info
curl http://localhost:8000/api/model/info

# Test history
curl http://localhost:8000/api/history
```

## 🧪 Testing dengan Python

```python
import requests

# Upload dan klasifikasi gambar
url = "http://localhost:8000/api/classify"
files = {"file": open("wereng_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🎓 Training Model

Gunakan notebook `model_training.ipynb` untuk melatih model Anda sendiri.

**Dataset yang direkomendasikan:**

- Wereng Coklat: min. 500 gambar
- Wereng Hijau: min. 500 gambar
- Wereng Punggung Putih: min. 500 gambar
- Bukan Wereng: min. 500 gambar

**Langkah-langkah:**

1. Download dataset dari Kaggle
2. Organize dataset ke folder sesuai class
3. Jalankan notebook `model_training.ipynb`
4. Simpan model hasil training ke `app/models/wereng_classifier.h5`
5. Restart server

## ⚙️ Mode Dummy Prediction

Jika model belum tersedia (`wereng_classifier.h5` tidak ada), API akan berjalan dalam mode **dummy prediction** untuk testing. Prediksi akan generate random namun konsisten untuk testing frontend.

Untuk menggunakan model real:

1. Train model menggunakan `model_training.ipynb`
2. Simpan model ke `app/models/wereng_classifier.h5`
3. Restart server

## 📝 Logging

Setiap prediksi akan dicatat di `logs/prediction_logs.txt`:

```
[2025-10-20 21:30:15] wereng_image.jpg → prediksi: Wereng Coklat (0.9400)
[2025-10-20 21:31:22] test_image.png → prediksi: Wereng Hijau (0.8750)
```

## 🔧 Konfigurasi

Edit `app/routes/info.py` untuk mengubah metadata model:

```python
MODEL_METADATA = {
    "model_name": "Wereng Classifier",
    "model_version": "1.0.0",
    "architecture": "MobileNetV2",
    ...
}
```

## 🐛 Troubleshooting

### Error: Model file not found

Model belum ada atau path salah. API akan berjalan dalam dummy mode.

**Solusi:** Train model dan simpan ke `app/models/wereng_classifier.h5`

### Error: TensorFlow not installed

TensorFlow belum terinstall dengan benar.

**Solusi:**

```bash
pip install tensorflow==2.15.0
# atau
pip install tensorflow-cpu==2.15.0
```

### Error: Port already in use

Port 8000 sudah digunakan aplikasi lain.

**Solusi:**

```bash
uvicorn app.main:app --reload --port 8001
```

## 📦 Deployment

### Menggunakan Gunicorn (Production)

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Menggunakan Docker (Coming Soon)

```dockerfile
# Dockerfile akan ditambahkan
```

## 🤝 Kontribusi

Proyek ini dibuat untuk membantu petani Indonesia mendeteksi hama wereng lebih awal.

## 📄 Lisensi

MIT License

## 👨‍💻 Author

Dibuat dengan ❤️ Ahmad Subadri (Abadbatok\_) untuk pertanian Indonesia

---

**Happy Coding! 🌾🚀**
