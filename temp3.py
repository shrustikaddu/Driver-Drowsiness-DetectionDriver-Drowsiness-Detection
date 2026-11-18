# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 23:01:45 2025

@author: HP
"""

import os
import sys

# Try importing OpenCV safely
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

def find_haar_cascade(filename_list):
    if not CV2_AVAILABLE:
        return None

    # 1) Current directory
    for fn in filename_list:
        if os.path.exists(fn):
            return fn

    # 2) cv2 package data directory
    datadir = getattr(cv2, 'data', None)
    if datadir:
        for fn in filename_list:
            path = os.path.join(datadir, fn)
            if os.path.exists(path):
                return path

    # 3) Common installation paths
    common_paths = [
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'cv2', 'data'),
        os.path.join(sys.prefix, 'share', 'opencv4', 'haarcascades'),
    ]
    for base in common_paths:
        for fn in filename_list:
            path = os.path.join(base, fn)
            if os.path.exists(path):
                return path

    return None

# List of Haar cascade files
face_cascades = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
]

# Exit gracefully if OpenCV is missing
if not CV2_AVAILABLE:
    print("OpenCV is not installed. Face detection is disabled, but the script runs safely.")
    sys.exit(0)

# Try to find Haar cascade file
face_cascade_path = find_haar_cascade(face_cascades)
if face_cascade_path is None:
    print("Could not find Haar cascade file. Please download one and place it in your project folder.")
    sys.exit(0)

print("Using Haar cascade:", face_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Cascade file is invalid or empty.")
    sys.exit(0)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    sys.exit(0)

print("Press Ctrl+C to stop capturing. Frames will be saved as 'frame_<count>.jpg'.")

frame_count = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save frame to file instead of showing window
        filename = f"frame_{frame_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        frame_count += 1

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and cleanup done.")
