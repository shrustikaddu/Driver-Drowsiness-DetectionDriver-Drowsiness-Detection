# Driver-Drowsiness-Detection

# Driver Drowsiness Detection System 🚗

## 📌 Project Overview

Driver Drowsiness Detection System is a Python-based computer vision project that monitors a driver's eye movements using a webcam.

The system uses OpenCV and MediaPipe Face Mesh to detect facial landmarks and calculate the Eye Aspect Ratio (EAR).

When the EAR remains below a predefined threshold for a certain number of consecutive frames, the system identifies a possible prolonged eye closure and displays a drowsiness alert with a sound notification.

## 🎯 Objectives

- Detect a face using a webcam.
- Identify eye landmarks in real time.
- Calculate the Eye Aspect Ratio (EAR).
- Monitor prolonged eye closure.
- Display a visual drowsiness warning.
- Generate an audio alert.

## ✨ Features

- Real-time webcam monitoring.
- Face landmark detection.
- Left and right eye tracking.
- Eye Aspect Ratio calculation.
- Automatic drowsiness warning.
- Sound alarm using Windows Winsound.
- Live EAR value display.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Image processing and webcam access |
| MediaPipe | Face and eye landmark detection |
| NumPy | Numerical operations |
| SciPy | Euclidean distance calculation |
| Winsound | Sound alert |

## ⚙️ Working Principle

1. The webcam captures live video.
2. OpenCV reads and processes each video frame.
3. MediaPipe Face Mesh detects facial landmarks.
4. The system identifies six landmarks around each eye.
5. Eye Aspect Ratio (EAR) is calculated for both eyes.
6. The average EAR is compared with a predefined threshold.
7. If the EAR stays below the threshold for 20 consecutive frames, a drowsiness alert is displayed.
8. A beep sound is generated.

## 📐 Eye Aspect Ratio (EAR)

The Eye Aspect Ratio is used to estimate whether the eyes are open or closed.

Formula:

EAR = (A + B) / (2 × C)

Where:

- A = Vertical distance between eye landmarks.
- B = Second vertical distance between eye landmarks.
- C = Horizontal distance between eye landmarks.

The average EAR of both eyes is calculated to monitor eye closure.

## 📂 Project Structure

```text
Driver-Drowsiness-Detection/
│
├── driver.py
├── README.md
└── requirements.txt
```

## 💻 Requirements

- Python 3.11
- Visual Studio Code
- Webcam
- Windows operating system

## 📦 Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/shrustikaddu/Driver-Drowsiness-Detection.git
```

### Step 2: Open the project folder

```bash
cd Driver-Drowsiness-Detection
```

### Step 3: Install required libraries

```bash
python -m pip install opencv-python mediapipe numpy scipy
```

### Step 4: Run the project

```bash
python driver.py
```

## 🔔 Alert System

The system uses the following parameters:

- EAR Threshold: 0.25
- Consecutive Frames: 20
- Alert Message: DROWSINESS ALERT!
- Sound Alert: Windows Winsound Beep

These are configurable example values and may require adjustment for different users and camera conditions.

## 🔮 Future Enhancements

- Improve detection under different lighting conditions.
- Add blink frequency analysis.
- Add head-position monitoring.
- Develop a graphical user interface.
- Add session reports and monitoring statistics.
- Explore support for additional operating systems.

## ⚠️ Limitations

- Requires a working webcam.
- Detection accuracy can vary with lighting and camera position.
- Glasses and facial movements may affect landmark detection.
- EAR alone cannot reliably determine whether someone is sleepy.
- This project is an educational prototype and is not a certified driving safety system.

## 👩‍💻 Author

**Srushti Kaddu**

GitHub: https://github.com/shrustikaddu

## ⭐ Acknowledgement

This project was developed as a learning project using Python, OpenCV, MediaPipe, NumPy and SciPy.

If you find this project useful, consider giving the repository a star!

