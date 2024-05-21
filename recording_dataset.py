import cv2
import mediapipe as mp
import time
import math
import csv
#  Standing, Fall, Sit, Bend Over
record = "Bend Over"

class poseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.pTime = 0
        self.recording = False  # Flag for recording data
        self.recorded_data = []  # List to store recorded data

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

    def startRecording(self):
        self.recording = True

    def stopRecording(self):
        self.recording = False

    def recordData(self, angle_1, angle_2, angle_3):
        if self.recording:
            self.recorded_data.append([angle_1, angle_2, angle_3, record])  # Label data as fall if recording

    def saveData(self, filename):
        
        # with open(filename, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['Angle_1', 'Angle_2', 'Angle_3', 'Label'])
        #     writer.writerows(self.recorded_data)

        # Append data to existing file
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.recorded_data)


def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(1)
    
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        detector.showFps(img)
        
        if len(lmList) >= 15:
            head = lmList[0]
            shoulder = lmList[11]
            hip = lmList[23]
            knee = lmList[25]
            
            angle_1 = math.degrees(math.atan2(shoulder[2] - head[2], shoulder[1] - head[1]))
            angle_2 = math.degrees(math.atan2(hip[2] - shoulder[2], hip[1] - shoulder[1]))
            angle_3 = math.degrees(math.atan2(knee[2] - hip[2], knee[1] - hip[1]))

            detector.recordData(angle_1, angle_2, angle_3)
            
            cv2.putText(img, f"Angle: {angle_1:.2f}, {angle_2:.2f}, {angle_3:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            
        cv2.imshow("Image", img)
        
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('q'):
            detector.saveData('falling_data.csv')
            break
        elif key == 27:  # Escape key
            break
        elif key == 32:  # Space bar
            if detector.recording:
                detector.stopRecording()
            else:
                detector.startRecording()


if __name__ == "__main__":
    main()
