
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import winsound

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
COUNTER = 0
alarm_triggered = False

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (600, 450))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            h, w, _ = frame.shape

            leftEye = np.array([
                [int(face.landmark[i].x * w),
                 int(face.landmark[i].y * h)]
                for i in LEFT_EYE
            ])

            rightEye = np.array([
                [int(face.landmark[i].x * w),
                 int(face.landmark[i].y * h)]
                for i in RIGHT_EYE
            ])

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

            if ear < EAR_THRESHOLD:
                COUNTER += 1

                if COUNTER >= EAR_CONSEC_FRAMES:
                    cv2.putText(
                        frame, "DROWSINESS ALERT!",
                        (100, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2
                    )

                    if not alarm_triggered:
                        winsound.Beep(2500, 700)
                        alarm_triggered = True
            else:
                COUNTER = 0
                alarm_triggered = False

            cv2.putText(
                frame, f"EAR: {ear:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2
            )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
face_mesh.close()
cv2.destroyAllWindows()
