# 🧠 Missing Person Detection System

A comprehensive **web-based application** that uses **AI-powered face recognition** to detect missing persons across CCTV camera networks in **real-time**.

---

## 🚀 Features

- **Real-time Face Recognition:** Advanced AI models for accurate face detection and matching  
- **CCTV Integration:** Support for RTSP streams and webcam feeds  
- **Web Interface:** User-friendly dashboard for managing persons and CCTV streams  
- **Live Detection:** Real-time face detection with bounding boxes and confidence scores  
- **Database Management:** JSON-based storage for persons, streams, and detection records  
- **RESTful API:** Complete API for integration with other systems  

---

## 🛠️ Technology Stack

- **Backend:** Flask (Python)  
- **Face Recognition:** InsightFace, OpenCV  
- **Frontend:** HTML5, Bootstrap, JavaScript  
- **Computer Vision:** OpenCV, NumPy  
- **Streaming:** RTSP, Webcam support  
- **Data Storage:** JSON files  

---

## 📋 Prerequisites

- Python 3.8+  
- Webcam (for testing)  
- CCTV cameras with RTSP support 

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd missing_person_system
```

### 2. Create Virtual Environment

```bash
conda create -n face_env python=3.10 -y
conda activate face_env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Required Packages:

```txt
flask==2.3.3
opencv-python==4.8.1.78
numpy==1.24.3
insightface==0.7.3
albumentations==1.3.1
torch==2.0.1
torchvision==0.15.2
python-dotenv==1.0.0
pillow==10.0.0
requests==2.31.0
werkzeug==2.3.7
onnxruntime==1.16.3
onnx==1.14.1
scipy==1.11.4
scikit-learn==1.3.2
tqdm==4.66.1
```

---

## 4. Environment Setup

Create a `.env` file:

```env
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
```

Generate a secret key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 5. Directory Structure

```bash
mkdir -p data/uploads/persons data/uploads/temp data/database
echo "{}" > data/database/persons.json
echo "{}" > data/database/cctv_streams.json
echo "[]" > data/database/detections.json
```

---

## 🚀 Usage

### 1. Start the Application

```bash
python app.py
```

### 2. Access the Web Interface

Open your browser and navigate to:  
👉 **http://localhost:8001**

---

## 3. System Workflow

### 🧍 Step 1: Register a Missing Person

- Go to **"Add Person"** page  
- Fill in person details:
  - Full Name  
  - Age  
  - Last Seen Location  
  - Contact Information  
  - Clear frontal face photo  
- Submit the form  
- System automatically extracts face embeddings  

---

### 🎥 Step 2: Configure CCTV Streams

- Go to **"CCTV Management"** page  
- Add CCTV streams:
  - Webcam: Use `"0"` as URL for local webcam  
  - RTSP: Use RTSP URLs for IP cameras  
  - Demo: Test streams for demonstration  

---

### 🔍 Step 3: Monitor Detection

- View **"Live Webcam"** stream in CCTV Management  
- System automatically detects faces in real-time  
- Registered persons are identified with:
  - ✅ Green bounding boxes  
  - 🧾 Name and confidence percentage  
  - ⚠️ Detection alerts  

---

## 📡 API Endpoints

### 👤 Person Management

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | `/person/upload` | Person registration form |
| POST | `/person/upload` | Register new person |
| GET | `/person/list` | List all registered persons |
| GET | `/person/test` | Config test endpoint |

### 🎥 CCTV Management

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | `/cctv/management` | CCTV management interface |
| GET | `/cctv/dashboard` | System dashboard |
| POST | `/cctv/add_stream` | Add new CCTV stream |
| GET | `/cctv/streams` | Get all stream status |
| GET | `/cctv/stream/<name>/frame` | Get stream frame |

### ⚙️ System API

| Method | Endpoint | Description |
|--------|-----------|-------------|
| GET | `/api/stats` | System statistics |
| GET | `/api/detections/recent` | Recent detections |
| GET | `/api/health` | Health check |

---

## 🔧 Configuration

### CCTV Stream URLs

- **Webcam:** `0` (default webcam)  
- **RTSP Example:** `rtsp://username:password@ip:port/stream`  
- **Demo Stream:** `demo`

### Face Recognition Settings

- Similarity Threshold: `0.6`  
- Face Quality Threshold: `0.7`  
- Model: `InsightFace Buffalo-L`

---

## 🎯 Key Components

### 1. Face Matcher (`models/face_matcher.py`)
- Advanced face embedding extraction  
- Multiple model support (InsightFace, DeepFace)  
- Face quality validation  
- Similarity comparison  

### 2. CCTV Manager (`models/cctv_manager.py`)
- RTSP stream handling  
- Webcam integration  
- Frame queue management  
- Stream health monitoring  

### 3. Web Interface (`templates/`)
- Responsive Bootstrap design  
- Real-time video streaming  
- Live detection alerts  
- System statistics  

---

## 🐛 Troubleshooting

### Webcam Not Working

```python
# Test webcam availability
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam available:', cap.isOpened())"
```

### RTSP Connection Failed

- Verify camera credentials  
- Check network connectivity  
- Test with VLC media player first  

### Face Detection Issues

- Use clear, frontal face photos  
- Ensure good lighting conditions  
- Check if face is clearly visible  

### Model Download Problems

- First run automatically downloads models (~100MB)  
- Ensure stable internet connection  
- Check: `/Users/username/.insightface/models/`

### Debug Endpoints

```text
http://localhost:8001/person/debug
http://localhost:8001/person/test
```

---

## 📊 System Architecture
```text
Frontend (Web App) → Flask API → AI Processing → CCTV Streams → Results
                      │
                      ├── Face Recognition
                      ├── Stream Management
                      ├── Database Storage
                      └── Real-time Detection
```
---

## 🔒 Security Considerations

- Store sensitive credentials securely  
- Use HTTPS in production  
- Implement authentication for CCTV access  
- Regular security updates  

---

## 🚀 Production Deployment

### 1. Environment Setup

```bash
export FLASK_ENV=production
export SECRET_KEY=your-production-secret
```

### 2. Use Production Server

```python
# Replace app.run() with:
from waitress import serve
serve(app, host='0.0.0.0', port=8001)
```

### 3. Database Migration

- Consider migrating from JSON files to **PostgreSQL**  
- Implement proper backup strategies  

---

## 📈 Monitoring

### Log Files

- Application logs: `app.log`  
- Access logs: Flask built-in  
- Error tracking: System alerts  

### Performance Metrics

- Face detection accuracy  
- Stream processing latency  
- System resource usage  

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch  

```bash
git checkout -b feature/AmazingFeature
```

3. Commit your changes  

```bash
git commit -m 'Add AmazingFeature'
```

4. Push to branch  

```bash
git push origin feature/AmazingFeature
```
5. Open a Pull Request  

---

## 📄 License

This project is licensed under the **MIT License** – see the LICENSE file for details.

---

## ❤️ Acknowledgments

- **InsightFace** team for face recognition models  
- **OpenCV** community for computer vision tools  
- **Flask** framework for web application foundation  

---

## 📞 Support

For support and questions:

- Check the troubleshooting section  
- Review system logs  
- Create a GitHub issue  
- Contact the development team  

---

> 🧩 **Note:** This system is designed for demonstration purposes.  
> For production use, implement additional security measures, proper database systems, and advanced error handling.

---
