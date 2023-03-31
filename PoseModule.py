import cv2 #OpenCV
import mediapipe as mp #Mediapipe for pose estimation
import time

class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth=True, detection_confidence = 0.5, tracking_confidence = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode = self.mode,
            model_complexity = self.complexity,
            smooth_landmarks = self.smooth,
            min_detection_confidence = self.detection_confidence,
            min_tracking_confidence = self.tracking_confidence)

    def find_pose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
            
        if draw and self.results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return img
        
    def get_points(self, img, draw=True):
        landmark_list = []

        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape # height,width,channel becoz in landmark x,y,z are in ratio of img sizes
                x, y = int(lm.x*w), int(lm.y*h)
                landmark_list.append([id,x,y])
                if draw:
                    cv2.circle(img, (x,y), 5, (0,255,0), cv2.FILLED)
            return landmark_list
    
def main():
    capture = cv2.VideoCapture('vids/v2.mp4')
    previous_time = 0

    detector = PoseDetector()

    while True:
        success, img = capture.read()

        img = detector.find_pose(img)

        position_list = detector.get_points(img)
        print(position_list)

        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time

        cv2.putText(img,str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()