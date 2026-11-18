# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 22:34:26 2025

@author: HP
"""
import cv2
import os
import sys

def find_haar_cascade(filename_list):
    """
    Try to find the Haar cascade file in a list of possible locations.
    Returns the full path if found, else None.
    """
    # 1) current directory
    for fn in filename_list:
        if os.path.exists(fn):
            return fn

    # 2) Try cv2 package data directory (if cv2.data exists)
    datadir = getattr(cv2, 'data', None)
    if datadir:
        for fn in filename_list:
            path = os.path.join(datadir, fn)
            if os.path.exists(path):
                return path

    # 3) Try environment paths / common install paths (you can add more if needed)
    # Example: Windows Anaconda
    common_paths = [
        os.path.join(sys.prefix, 'Lib', 'site-packages', 'cv2', 'data'),  # typical conda + opencv‑python install
        os.path.join(sys.prefix, 'share', 'opencv4', 'haarcascades'),     # typical linux install
    ]
    for base in common_paths:
        for fn in filename_list:
            path = os.path.join(base, fn)
            if os.path.exists(path):
                return path

    # Not found
    return None

# List of filenames to try for face cascade (most common first)
face_cascades = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    # you can add more variants if you downloaded others
]

face_cascade_path = find_haar_cascade(face_cascades)
if face_cascade_path is None:
    print("Error: Could not find any of the Haar cascade files for face detection.")
    print("Please download one of the following and place it in your project folder or ensure it is accessible:")
    print(face_cascades)
    sys.exit(1)

print("Using face cascade file:", face_cascade_path)
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Error: Loaded cascade is empty. Please verify the XML file at:", face_cascade_path)
    sys.exit(1)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
