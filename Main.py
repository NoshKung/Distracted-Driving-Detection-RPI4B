import os
import cv2
import mediapipe as mp
import time
from pydub import AudioSegment
from pydub.playback import play
import utils
import math
import numpy as np
import RPi.GPIO as GPIO

# Set environment variable
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# Initialize variables
CLOSED_EYES_FRAME = 30
FONTS = cv2.FONT_HERSHEY_COMPLEX
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# GPIO configuration
BUTTON_PIN = 17  # Change this to the GPIO pin you are using
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def is_detection_enabled():
    return GPIO.input(BUTTON_PIN) == GPIO.HIGH

def landmarks_detection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

def euclidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def blink_ratio(img, landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rh_distance = euclidean_distance(rh_right, rh_left)
    rv_distance = euclidean_distance(rv_top, rv_bottom)
    lv_distance = euclidean_distance(lv_top, lv_bottom)
    lh_distance = euclidean_distance(lh_right, lh_left)

    re_ratio = rh_distance / rv_distance
    le_ratio = lh_distance / lv_distance

    ratio = (re_ratio + le_ratio) / 2
    return ratio

def head_pose_detection(frame, face_mesh):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    h, w, c = frame.shape
    head_direction = ""
    
    if results.multi_face_landmarks:
        for face_lms in results.multi_face_landmarks:
            point137 = face_lms.landmark[137]
            cx137, cy137 = int(point137.x * w), int(point137.y * h)
            cv2.circle(frame, (cx137, cy137), 10, (0, 0, 255), -1)

            point4 = face_lms.landmark[4]
            cx4, cy4 = int(point4.x * w), int(point4.y * h)
            cv2.circle(frame, (cx4, cy4), 10, (0, 255, 0), -1)

            point366 = face_lms.landmark[366]
            cx366, cy366 = int(point366.x * w), int(point366.y * h)
            cv2.circle(frame, (cx366, cy366), 10, (255, 0, 0), -1)

            cv2.line(frame, (cx4, cy4), (cx137, cy137), (0, 0, 255), 5)
            cv2.line(frame, (cx4, cy4), (cx366, cy366), (255, 0, 0), 5)

            distL = int(cx4 - cx137)
            distR = int(cx366 - cx4)

            if (distL > distR) and (distL - distR) > 40:
                head_direction = "Right"
            elif (distR > distL) and (distR - distL) > 40:
                head_direction = "Left"
            else:
                head_direction = "Straight"

    return head_direction

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Initialize face mesh model
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

    # Load audio alerts
    song1 = AudioSegment.from_mp3("1.mp3")
    song2 = AudioSegment.from_mp3("2.mp3")

    # Initialize variables
    frame_counter = 0
    CEF_COUNTER = 0
    start_time = time.time()
    notification_displayed = False
    turned_head = False
    current_head_direction = ""

    while cap.isOpened():
        frame_counter += 1
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Check for key press
        key = cv2.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

        if not is_detection_enabled():
            # Skip frame processing if detection is disabled
            continue

        # Head pose detection
        current_head_direction = head_pose_detection(frame, face_mesh)

        if current_head_direction == "Right":
            if not turned_head:
                start_time = time.time()
                turned_head = True
        elif current_head_direction == "Left":
            if not turned_head:
                start_time = time.time()
                turned_head = True
        else:
            turned_head = False

        elapsed_time = time.time() - start_time

        if turned_head and elapsed_time > 5:
            play(song1)
            # Reset the timer to avoid continuous playback
            start_time = time.time()
            turned_head = False  # Optionally, reset the turned_head flag

        # Eye blink detection
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarks_detection(frame, results, False)
            ratio = blink_ratio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

            utils.colorBackgroundText(frame, f'Ratio: {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.WHITE, utils.RED)

            if ratio > 5:
                CEF_COUNTER += 1

                if CEF_COUNTER <= 3:
                    utils.colorBackgroundText(frame, f'1', FONTS, 5, (50, 300), 6, utils.WHITE, utils.BLUE, pad_x=4, pad_y=6)
                elif CEF_COUNTER <= 20:
                    utils.colorBackgroundText(frame, f'2', FONTS, 5, (50, 300), 6, utils.WHITE, utils.BLUE, pad_x=4, pad_y=6)
                elif CEF_COUNTER <= CLOSED_EYES_FRAME:
                    utils.colorBackgroundText(frame, f'3', FONTS, 5, (50, 300), 6, utils.WHITE, utils.BLUE, pad_x=4, pad_y=6)
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    play(song2)
                    CEF_COUNTER = 0
            else:
                CEF_COUNTER = 0

            cv2.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.BLUE, 1, cv2.LINE_AA)
            cv2.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.BLUE, 1, cv2.LINE_AA)

        # Display FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time
        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)

        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('DistractedDrivingDetection', frame)

    cv2.destroyAllWindows()
    cap.release()
    GPIO.cleanup()

if __name__ == "__main__":
    main()
