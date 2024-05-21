import cv2
import base64
import numpy as np
import requests
import json

# Server URL
SERVER_URL = 'http://127.0.0.1:5000/detect_fall'
API_KEY = '9rqKxPXzYNCEsrFrwRhDzIL7UgM3ld45dEF7W7KmmLe'

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    base64_encoded_frame = base64.b64encode(buffer).decode('utf-8')

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Prepare data to send to the server
    data = {'frame': base64_encoded_frame, 'api_key': API_KEY}
    headers = {'Content-Type': 'application/json'}

    # Send frame to server
    response = requests.post(SERVER_URL, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        print(response_data)
    else:
        print('Error:', response.status_code)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
