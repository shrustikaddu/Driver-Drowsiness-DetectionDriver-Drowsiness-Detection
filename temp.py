import pygame
import pygame.camera
from PIL import Image
import numpy as np
import dlib
from scipy.spatial import distance as dist
import threading
import time

# initialize pygame camera
pygame.camera.init()
cam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
cam.start()

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

try:
    while True:
        img = cam.get_image()  # pygame Surface
        arr = pygame.surfarray.array3d(img)  # width x height x RGB
        # convert to grayscale manually
        gray = np.dot(arr[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            coords = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            leftEye  = coords[lStart:lEnd]
            rightEye = coords[rStart:rEnd]
            ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        # sound alarm or print warning
                        print("DROWSINESS ALERT!")
            else:
                COUNTER = 0
                ALARM_ON = False

        time.sleep(0.03)  # small delay to reduce CPU usage

except KeyboardInterrupt:
    cam.stop()