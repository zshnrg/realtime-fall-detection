import os
import tempfile
import joblib
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import math
import numpy as np
import base64
import requests

app = Flask(__name__)


class poseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

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
                self.mpDraw.draw_landmarks(
                    img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def getPosition(self, img):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        return lmList


def send_line_notify(message, img_path, token):
    url = "https://notify-api.line.me/api/notify"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    payload = {
        'message': message
    }

    try:
        with open(img_path, 'rb') as file:
            files = {'imageFile': file}
            response = requests.post(
                url, headers=headers, params=payload, files=files)
    except Exception as e:
        print(f"An error occurred while sending the image: {e}")
    return response.status_code


# Load the model and pose detector once when the app starts
model_path = os.path.join(os.path.dirname(
    __file__), 'fall_detection_model.pkl')
model = joblib.load(model_path)
detector = poseDetector()


@app.route('/detect_fall', methods=['POST'])
def detect_fall():
    data = request.json

    if 'api_key' not in data:
        return jsonify({'error': 'LINE API token is missing'}), 401
    
    if 'location' not in data:
        return ({'error': 'Camera location is missing'}), 401

    bot_token = data['api_key']
    location = data['location']

    base64_image = data.get('frame')
    image_data = base64.b64decode(base64_image)

    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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

        angle_1 = math.degrees(math.atan2(
            shoulder[2] - head[2], shoulder[1] - head[1]))
        angle_2 = math.degrees(math.atan2(
            hip[2] - shoulder[2], hip[1] - shoulder[1]))
        angle_3 = math.degrees(math.atan2(knee[2] - hip[2], knee[1] - hip[1]))

        input_data = [[angle_1, angle_2, angle_3]]
        prediction = model.predict(input_data)

        response["prediction"] = prediction[0]
        response["angles"] = [angle_1, angle_2, angle_3]

        if prediction[0] == 'Fall':
            print("Fall Detected")
            _, img_path = tempfile.mkstemp(suffix='.jpg')
            cv2.imwrite(img_path, img)
            message = "Alert: Someone has fallen!" + f'\n\nLocation: {location}' + '\nEmergency Number: *119*'
            send_line_notify(message, img_path, bot_token)

    return jsonify(response)


if __name__ == '__main__':
    app.run(threaded=True)
