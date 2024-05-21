import os
import joblib
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import base64
import requests

app = Flask(__name__)

# Detect fall using the trained model
# Send the result to the user on LINE Notify

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

def send_line_notify(message, img, line_notify_token):
    url = "https://notify-api.line.me/api/notify"
    # Post the message to LINE Notify
    headers = {
        'Authorization': f'Bearer {line_notify_token}',
        'Content-Type': 'multipart/form-data'
    }
    payload = {
        'message': message,
        'imageFile': base64.b64encode(img).decode('utf-8')
    }
    response = requests.post(url, headers=headers, data=payload)
                     

@app.route('/detect_fall', methods=['POST'])
def detect_fall():
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if bot token is provided
    bot_token = request.args.get('bot_token')
    if not bot_token:
        return jsonify({"error": "No bot token provided"}), 400
    
    print(f"Received image and LINE Notify with bot token: {bot_token}")
    
    image_data = file.read()
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Get the user's chat ID
    chat_id = request.json.get('message', {}).get('chat', {}).get('id')
    if not chat_id:
        return jsonify({"error": "Chat ID not found in request"}), 400

    detector = poseDetector()
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    
    response = {
        "prediction": "No fall detected",
        "angles": []
    }
    
    if len(lmList) >= 15:
        head = lmList[0]
        shoulder = lmList[11]
        hip = lmList[23]
        knee = lmList[25]

        angle_1 = math.degrees(math.atan2(shoulder[2] - head[2], shoulder[1] - head[1]))
        angle_2 = math.degrees(math.atan2(hip[2] - shoulder[2], hip[1] - shoulder[1]))
        angle_3 = math.degrees(math.atan2(knee[2] - hip[2], knee[1] - hip[1]))

        input_data = [[angle_1, angle_2, angle_3]]

        # Load the model from the correct path
        model_path = os.path.join(os.path.dirname(__file__), 'fall_detection_model.pkl')
        model = joblib.load(model_path)
        prediction = model.predict(input_data)

        response["prediction"] = prediction[0]
        response["angles"] = [angle_1, angle_2, angle_3]

        if prediction[0] == 'fall':
            # Send a LINE Notify message with the image
            line_notify_token = bot_token
            send_line_notify("Fall detected!", image_data, line_notify_token)

    return jsonify(response)

if __name__ == '__main__':
    app.run()