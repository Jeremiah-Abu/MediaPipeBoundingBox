#https://stackoverflow.com/questions/66876906/create-a-rectangle-around-all-the-points-returned-from-mediapipe-hand-landmark-d
import cv2
import time
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0) 

_, frame = cap.read()

h, w, c = frame.shape #determine height width and channel of cam

# used to record the time when we processed last frame
prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0
 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame,1) #Mirror screen
        
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        result = hands.process(framergb)
        black = np.zeros((h,w,c), np.uint8) #create black image with same dimentions as webcam
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                cv2.rectangle(black, (x_min-15, y_min-15), (x_max+15, y_max+15), (0, 255, 0), 2) #display bounding box arround and
                mp_drawing.draw_landmarks(
                        black,
                        handLMs,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps) #convert fps to integer
        fps = str(fps) #convert fps to string
        cv2.putText(black, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) #print fps
        
        cv2.imshow("Frame", black)

        cv2.waitKey(1)
