# PPE Detection Using YOLOv8

A real-time **Personal Protective Equipment (PPE) Detection System** developed during my internship at **Rashtriya Ispat Nigam Limited (RINL), Visakhapatnam Steel Plant**.

The project uses **YOLOv8 (Ultralytics)**, a deep learning-based object detection model, to identify safety equipment and PPE violations in images and real-time webcam streams.

---

## 📌 Project Overview

In industrial environments, ensuring that workers consistently wear appropriate Personal Protective Equipment is critical for workplace safety.

Traditional PPE compliance monitoring relies heavily on manual supervision, which can be difficult to scale across large industrial sites and may not provide continuous real-time monitoring.

This project explores an automated computer vision approach using **YOLOv8** to detect PPE equipment and identify potential safety violations.

The system supports:

* 🖼️ Image-based PPE detection
* 📹 Real-time webcam detection
* 🌐 Web-based inference through a Flask backend
* 📦 Custom dataset preparation and annotation
* 🧠 YOLOv8 model training and evaluation
* ⚙️ Real-time inference using OpenCV

---

## 🎯 Objectives

The main objectives of the project were to:

1. Develop an object detection model using YOLOv8.
2. Prepare and train the model on a custom PPE dataset.
3. Detect both the presence and absence of required safety equipment.
4. Evaluate model performance across different PPE classes.
5. Build a web interface for image-based inference.
6. Integrate the trained model with a real-time webcam stream.
7. Understand the complete machine learning workflow from **dataset preparation → training → evaluation → deployment → inference**.

---

## 🧠 Model

### YOLOv8

The project uses **YOLOv8m (YOLOv8 Medium)** from the Ultralytics framework.

YOLO (You Only Look Once) performs object detection by predicting object locations and classes within an image.

The model was used for:

* Bounding box prediction
* Object classification
* Confidence estimation
* Real-time inference

**Model:** `YOLOv8m.pt`
**Input resolution:** `640 × 640`
**Confidence threshold:** `0.3`

---

## 📊 Dataset

A custom annotated PPE dataset was used for training.

The dataset contains **11 classes**, covering both PPE equipment and safety violations:

| Class ID | Class         |
| -------: | ------------- |
|        0 | Safety Helmet |
|        1 | Safety Gloves |
|        2 | Safety Vest   |
|        3 | Safety Boots  |
|        4 | Goggles       |
|        5 | Person        |
|        6 | No Helmet     |
|        7 | No Gloves     |
|        8 | No Vest       |
|        9 | No Boots      |
|       10 | No Goggles    |

The dataset was organized in the standard YOLO format:

```text
combined_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

Each annotation contains:

```text
class_id x_center y_center width height
```

with bounding-box coordinates normalized between 0 and 1.

The dataset contained approximately:

* **800 training images**
* **100 validation images**
* **100 test images**
* **4000+ training annotations**

Image resolution varied, with training performed using YOLOv8's preprocessing at `640 × 640`.

---

## 🔄 Machine Learning Pipeline

```text
Raw Images
     ↓
Dataset Annotation
     ↓
Dataset Validation & Cleaning
     ↓
YOLO-format Dataset
     ↓
Train / Validation / Test Split
     ↓
YOLOv8 Model Training
     ↓
Model Evaluation
     ↓
Performance Tuning
     ↓
Trained Model
     ↓
 ┌───────────────┬─────────────────┐
 ↓               ↓
Web Application  Real-Time Webcam
(Flask)          (OpenCV)
```

---

## 🛠️ Technologies Used

### Machine Learning / Deep Learning

* YOLOv8
* Ultralytics
* Deep Learning
* Object Detection

### Programming

* Python

### Computer Vision

* OpenCV
* Pillow

### Backend

* Flask
* Waitress

### Frontend

* HTML
* CSS
* JavaScript

### Development & Training

* Google Colab
* Google Drive
* Anaconda / Conda
* VS Code

---

## 🌐 Web Application

A lightweight web application was developed to make the trained model accessible through a browser.

### Workflow

```text
User uploads image
       ↓
Frontend sends image to Flask
       ↓
Flask receives image
       ↓
YOLOv8 performs inference
       ↓
Detection results returned
       ↓
Frontend renders bounding boxes
       ↓
Detected PPE displayed to user
```

The Flask backend exposes a `/detect` endpoint that receives uploaded images and returns detection results.

Example detection output:

```text
[x1, y1, x2, y2, "helmet", 0.89]
[x1, y1, x2, y2, "gloves", 0.67]
```

---

## 📹 Real-Time Webcam Detection

The trained model was also integrated with **OpenCV** for real-time detection.

Each webcam frame is:

1. Captured using OpenCV.
2. Passed to YOLOv8.
3. Processed for object detection.
4. Annotated with bounding boxes and labels.
5. Displayed as a live video stream.

Simplified workflow:

```text
Webcam
  ↓
Frame Capture
  ↓
YOLOv8 Inference
  ↓
Object Detection
  ↓
Bounding Box + Label
  ↓
Live Display
```

The system achieved approximately **10–15 FPS on a mid-range laptop** during the reported testing configuration.

---

## 📈 Results

The trained YOLOv8 model was evaluated qualitatively across the 11 PPE classes.

### Stronger Detection

* Safety helmets were detected consistently.
* The `person` class showed very high detection performance.
* Safety boots were detected reliably in visible cases.

### Areas Requiring Improvement

Performance was comparatively weaker for:

* Safety gloves
* Safety vests
* Goggles
* Rare PPE violation classes

Some errors occurred when PPE objects overlapped with people or other equipment, or when objects were small or affected by lighting conditions.

These observations highlighted the importance of **dataset quality, class balance, annotation accuracy, and appropriate image resolution** in object detection.

---

## 🧪 Challenges & Problem Solving

During development, several practical machine learning and software engineering challenges were encountered.

### 1. Annotation and Overlapping Objects

Some images contained overlapping PPE objects, such as gloves and vests.

**Impact:**

* Difficult label separation
* Confusion between classes
* Reduced detection reliability for small objects

**Learning:**
Annotation quality has a direct impact on model performance.

---

### 2. YAML Configuration

The dataset initially contained Windows-specific paths that caused problems when training through Google Colab.

The configuration was modified to use relative paths:

```yaml
train: images/train
val: images/val
test: images/test
```

---

### 3. Model Inference and Frontend Integration

Incorrect bounding-box structures and frontend rendering issues initially resulted in incorrect or overlapping labels.

The issue was addressed by simplifying the frontend rendering and using YOLOv8's built-in result plotting functionality.

---

### 4. Environment and Dependency Issues

Different Python environments caused package-related errors such as:

```text
ModuleNotFoundError: ultralytics
```

The issue was resolved by identifying the active Python environment, installing dependencies in the correct environment, and configuring the VS Code interpreter accordingly.

---

### 5. Real-Time Inference Performance

The YOLOv8m model was computationally demanding on systems without a dedicated GPU.

Testing showed that inference latency could become significant on lower-end hardware.

Potential solutions include:

* Using YOLOv8n or YOLOv8s
* Model optimization
* GPU acceleration
* ONNX/TensorRT deployment
* Quantization

---

## 🚀 Future Improvements

Potential extensions of this project include:

### Model Optimization

* Experiment with different YOLOv8 model sizes.
* Increase dataset size and class balance.
* Apply class-specific augmentation.
* Optimize inference using ONNX or TensorRT.
* Explore quantization for edge deployment.

### Real-Time Safety Monitoring

* Integrate CCTV/video-stream processing.
* Add automated alerts for PPE violations.
* Track individuals across frames.
* Support multiple camera feeds.

### Deployment

* Deploy the application to a cloud platform.
* Build a responsive mobile interface.
* Explore edge deployment using devices such as Raspberry Pi or NVIDIA Jetson.

### Analytics

* Store detection results.
* Generate safety compliance statistics.
* Build dashboards for monitoring PPE violations.
* Generate automated safety reports.

---

## 💡 Key Learning Outcomes

This project provided hands-on experience with the complete lifecycle of a Deep Learning computer vision application:

* Dataset preparation and annotation
* YOLO-format datasets
* Deep Learning model training
* Object detection
* Model evaluation
* Confidence thresholds
* Debugging model and data issues
* Python-based inference
* Computer vision with OpenCV
* Flask-based model serving
* Frontend-backend integration
* Real-time inference
* Environment and dependency management
* Performance optimization

Most importantly, the project demonstrated that building an ML system involves much more than training a model—it requires **data preparation, experimentation, debugging, evaluation, deployment, and iterative improvement**.

---

## 📂 Suggested Project Structure

```text
PPE-Detection-YOLOv8/
│
├── dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── models/
│   └── best.pt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── object_detector.py
├── realtime_detector.py
├── requirements.txt
└── README.md
```

> **Note:** The exact file/folder structure may differ depending on the version of the project uploaded to this repository.

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Create and activate a Python environment:

```bash
conda create -n ppe-detection python=3.10
conda activate ppe-detection
```

Install dependencies:

```bash
pip install ultralytics opencv-python flask waitress pillow
```

---

## ▶️ Running the Project

### Image Detection Web Application

Run the Flask application:

```bash
python object_detector.py
```

Then open:

```text
http://localhost:8080
```

Upload an image to view detected PPE classes and bounding boxes.

### Real-Time Webcam Detection

Run:

```bash
python realtime_detector.py
```

The webcam feed will be processed frame-by-frame using the trained YOLOv8 model.

Press **Q** to exit.

---

## 📌 Project Context

This project was completed as part of my internship at:

**Rashtriya Ispat Nigam Limited (RINL) – Visakhapatnam Steel Plant**

**Duration:** 23 June 2025 – 19 July 2025

The project explored the application of Deep Learning and computer vision to industrial safety monitoring.

---

## 👩‍💻 Author

**Sri Sowmya Jahnavi Karedla**

Computer Science & Engineering
Andhra University College of Engineering

* GitHub: `https://github.com/YOUR-USERNAME`
* LinkedIn: `YOUR-LINKEDIN-PROFILE`

---

## ⭐ Acknowledgements

* **Rashtriya Ispat Nigam Limited (RINL), Visakhapatnam Steel Plant**
* **Ultralytics YOLOv8**
* OpenCV
* Flask
* Google Colab

---

## 📜 License

This project is intended primarily for educational and research purposes.

Please refer to the repository contents for any additional licensing information.
