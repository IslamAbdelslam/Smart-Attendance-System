# 🎓 Smart Attendance System

> **Automated Face Recognition Attendance using YOLOv8 + ArcFace**

Graduation Project , Faculty of Computers and Information Science, Mansoura University 2026

🌐 **Live Demo:** [huggingface.co/spaces/Haneen13/smart-attendance-system](https://huggingface.co/spaces/Haneen13/smart-attendance-system)

---

## 🔍 Overview

An AI-powered system that automatically marks student attendance by recognizing all faces in a classroom frame simultaneously. No manual roll call.

The project has two parts:

🖥️ **Local Script** runs on a laptop connected to a camera. It detects all faces at once, matches them against an enrolled database, and logs attendance to CSV.

🌐 **Web Application** (`Smart_Attendence_system_APP/`) is a Flask website deployed on Hugging Face Spaces with a live demo, enrollment interface, and REST API.

---

## ⚙️ Pipeline

```
Camera Frame
    ↓  YOLOv8s-Face detects all faces simultaneously
    ↓  Crop and resize to 112×112
    ↓  ArcFace w600k_mbf extracts 512-dim identity vector
    ↓  Cosine similarity match against enrolled database
    →  Mark present and log to CSV
```


---

## 📁 Project Structure

```
Smart-Attendance-System/
│
├──  Smart_Attendence_system_APP/    Web application (Hugging Face)
│   ├── templates/index.html           Website with embedded CSS and JS
│   ├── app.py                         Flask routes and REST API
│   ├── config.py                      Web app settings
│   ├── face_pipeline.py               AI pipeline for web
│   ├── Dockerfile                     Container for Hugging Face deployment
│   └── requirements.txt               Web app dependencies
│
├──  NoteBook/                       Google Colab preprocessing pipeline
│   └── Smart_attendence_system.ipynb
│
├──  config.py                       Local script settings
├──  face_pipeline.py                AI pipeline functions
├──  main.py                         Real-time camera attendance loop
├──  enroll.py                       Enroll new students (5-capture flow)
├──  manage_database.py              Database management tool
├──  requirements.txt                Local script dependencies
└──  README.md
```

> ⚠️ **Not in repo:** `Models/` (download separately), `database/` (private student data), `logs/` (private attendance records)

---

## 🚀 Installation — Local Script

**Requirements:** Python 3.11, webcam or IoT camera

```bash
git clone https://github.com/Haneenmohammed1311/Smart-Attendance-System.git
cd Smart-Attendance-System

py -3.11 -m venv venv
venv\Scripts\activate          # Windows


pip install -r requirements.txt
```

> 💡 If insightface fails: `pip install insightface --only-binary=:all:`

### 📥 Download Model Weights

```
🔷 YOLOv8s-Face (21 MB)
   Source  : https://github.com/lindevs/yolov8-face/releases
   Save to : models/yolov8s-face-lindevs.pt

🔷 ArcFace w600k_mbf (65 MB)
   Source  : https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip
   Extract : w600k_mbf.onnx
   Save to : models/w600k_mbf.onnx
```

---

## 🎯 Usage — Local Script

**👤 Enroll students**
```bash
python enroll.py
```
Enter name, capture 5 photos with quality checks. Each capture is verified before being accepted just like Face ID.

**📸 Run attendance**
```bash
python main.py
```
All faces in the frame are recognized simultaneously. Color code on screen: 
🟢 Green means just marked present, 
🟡 Yellow means already marked, 
🔴 Red means unknown. Press Q to quit and see session summary.

**🗃️ Manage database**
```bash
python manage_database.py
```

**🔧 Switch camera source** edit `config.py`:
```python
CAMERA_SOURCE = 0                              # Laptop webcam
CAMERA_SOURCE = "http://192.168.1.100/video"   # IoT camera via WiFi
```

---

## 🌐 Installation — Web Application

```bash
cd Smart_Attendence_system_APP

py -3.11 -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python app.py
# Open: http://localhost:7860
```

---

## 🔌 REST API

| Method | Endpoint | Description |
|---|---|---|
| POST | /recognize | Upload frame, detect all faces, return names and annotated image |
| POST | /enroll/capture | One capture per call, call 5 times with quality check |
| POST | /enroll/save | Average 5 embeddings and save to database |
| POST | /enroll/cancel | Clear incomplete enrollment buffer |
| GET | /attendance | Get today's attendance log |
| GET | /database | List all enrolled students |
| DELETE | /database/name | Remove a student from database |

---

## 🛠️ Technologies

| Component | Tool |
|---|---|
| 🎯 Face Detection | YOLOv8s-Face (lindevs, 2023) |
| ✂️ Face Alignment | insightface buffalo_sc |
| 🧠 Face Embedding | ArcFace w600k_mbf (CVPR 2019) |
| 🔍 Similarity | Cosine Similarity |
| 📷 Camera | OpenCV VideoCapture |
| ⚡ Runtime | ONNX Runtime (CPU) |
| 🌐 Web Framework | Flask |
| 🚀 Deployment | Hugging Face Spaces + Docker |

---

## 📦 Deployment on Hugging Face

Flask on Hugging Face Spaces requires Docker. The `Dockerfile` is included in `Smart_Attendence_system_APP/`.

The app listens on port 7860. Model files are uploaded separately via the Hugging Face Files tab using Git LFS.

---

## 📚 Key Papers

| Paper | Authors | Venue |
|---|---|---|
| ArcFace: Additive Angular Margin Loss | Deng et al. | CVPR 2019 |
| YOLOv8 | Jocher, Chaurasia, Qiu | Ultralytics 2023 |

---


## 📄 License

Academic and research use only.
Model weights follow the [InsightFace non-commercial license](https://github.com/deepinsight/insightface).