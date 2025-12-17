# 🧠 Missing Person Detection System

A comprehensive **web-based application** that uses **AI-powered face recognition** to detect missing persons across CCTV camera networks in **real-time**.

![UI Screenshot](https://via.placeholder.com/800x400?text=Missing+Person+Detection+System+Dashboard)

---

## 🚀 Features

- **Real-time Face Recognition:** Advanced AI models (InsightFace) for accurate detection even in crowded or low-quality streams.
- **Modern Dark UI:** Professional "Glassmorphism" design with live heartbeat animations and dark mode for low-light monitoring.
- **CCTV Integration:** seamless support for multiple RTSP streams and webcams.
- **Live Detection:** Real-time bounding boxes, confidence scores, and audible/visual alerts.
- **Database Management:** **SQLite** database for robust storage of persons, streams, and detection history.
- **Modular Architecture:** Enterprise-grade flask structure using Blueprints and Application Factory pattern.
- **RESTful API:** Complete API for integration with other systems.

---

## 🛠️ Technology Stack

- **Backend:** Flask (Python) with Blueprints
- **Face Recognition:** InsightFace (Buffalo_L model), OpenCV
- **Database:** SQLite (SQLAlchemy ORM)
- **Frontend:** HTML5, Bootstrap 5, CSS3 (Glassmorphism), JavaScript
- **Computer Vision:** OpenCV, NumPy, Albumentations
- **Streaming:** Multi-threaded RTSP/Webcam processing

---

## 📋 Prerequisites

- Python 3.9+
- Webcam (for local testing)
- IP Cameras (RTSP) for production usage

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd missing_person_system
```

### 2. Create Virtual Environment

```bash
# MacOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

You need to install the required system libraries for OpenCV and InsightFace.

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Start the Application

The application uses a modular structure. Start it using the entry point:

```bash
python run.py
```

### 2. Access the Web Interface

Open your browser and navigate to:
👉 **http://localhost:8001**

---

## 5. Directory Structure

The project follows a modular "Application Factory" pattern:

```text
face_detection/
├── app/
│   ├── __init__.py         # App factory & DB creation
│   ├── models/             # Database & ML Models
│   │   ├── db_models.py    # SQLAlchemy Tables
│   │   ├── cctv_manager.py # Stream Logic
│   │   └── face_matcher.py # AI Logic
│   ├── routes/             # Blueprints
│   ├── static/             # CSS/JS Assets
│   └── templates/          # HTML Templates
├── data/                   # SQLite DB & Uploads
├── run.py                  # Entry Point
├── config.py               # Config definition
└── requirements.txt
```

---

## 3. System Workflow

### 🧍 Step 1: Register a Missing Person
- Go to **"Add Person"** page.
- Upload a clear photo.
- The system automatically extracts 512D face embeddings and stores them in SQLite.

### 🎥 Step 2: Configure CCTV Streams
- Go to **"CCTV Management"**.
- Add RTSP URLs (e.g., `rtsp://admin:pass@192.168.1.10:554/stream`).
- Or use `0` for the local webcam.
- Toggle streams On/Off instantly.

### 🔍 Step 3: Monitor Detection
- Watch the **Dashboard** or **CCTV View**.
- **Green Box**: Match Found (Missing Person Detected).
- **Red Box**: Unknown/Scanning.
- All detections are logged to the database with timestamps and location.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | `/api/persons` | List all registered persons |
| GET | `/api/stats` | System live statistics |
| GET | `/api/detections/recent` | Recent detections list |
| POST | `/api/search` | Search person by image upload |
| GET | `/api/health` | System health check |

---

## 🔧 Configuration

### CCTV Stream URLs
- **Webcam:** `0` (default)
- **RTSP:** `rtsp://user:password@ip:port/path`

### Face Recognition Settings
- **Threshold:** `0.5` (Adjust in `app/models/face_matcher.py`)
- **Model:** `buffalo_l` (InsightFace)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## ❤️ Acknowledgments

- **InsightFace** for state-of-the-art face analysis.
- **Flask** community for the robust backend framework.

