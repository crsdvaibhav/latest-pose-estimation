import cv2 #OpenCV
import mediapipe as mp #Mediapipe for pose estimation
import time

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
"""
Params -> 
static_image_mode = False, first will detect till a detection confidence then goto tracking
upper_body_only = False, 25 instead of 33 landmarks
smooth_landmarks = True
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
"""

capture = cv2.VideoCapture('vids/v2.mp4')
previous_time = 0

while True:
    success, img = capture.read()

    #Converting our img to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    # Printing and drawing landmarks
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape # height,width,channel becoz in landmark x,y,z are in ratio of img sizes
            print(id, lm)
            x, y = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (x,y), 5, (0,255,0), cv2.FILLED)
        
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    cv2.putText(img,str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)