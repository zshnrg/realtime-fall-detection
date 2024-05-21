import cv2
import mediapipe as mp
import time
import math
import joblib

class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                      smooth_landmarks=self.smooth,
                                      min_detection_confidence=self.detectionCon,
                                      min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def showFps(self, img):
        cTime = time.time()
        fbs = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, "FPS: " + str(int(fbs)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 2)

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(1)

    # Load the trained model
    model = joblib.load('fall_detection_model.pkl')

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        detector.showFps(img)
        
        # Calculate body angle from head, shoulder, hip, and knee landmarks
        if len(lmList) >= 15:  # Ensure required landmarks are detected
            head = lmList[0]  # Landmark for the head
            shoulder = lmList[11]  # Landmark for the shoulder
            hip = lmList[23]  # Landmark for the hip
            knee = lmList[25]  # Landmark for the knee
            
            # Calculate angle using the arctangent function
            angle_1 = math.degrees(math.atan2(shoulder[2] - head[2], shoulder[1] - head[1]))
            angle_2 = math.degrees(math.atan2(hip[2] - shoulder[2], hip[1] - shoulder[1]))
            angle_3 = math.degrees(math.atan2(knee[2] - hip[2], knee[1] - hip[1]))

            # Preprocess the angle data
            input_data = [[angle_1, angle_2, angle_3]]

            # Make predictions using the loaded model
            prediction = model.predict(input_data)

            # Display the prediction results
            cv2.putText(img, prediction[0], (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
        
        cv2.imshow("Image", img)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
