"""
testSim.py
-----------
Runs the trained model LIVE against the Udacity simulator's
"Autonomous Mode". Opens a socket server; simulator sends live
camera frames + speed, this script predicts steering angle and
sends throttle+steering back to the simulator in real time.

Run:
    python testSim.py
Then open the Udacity Simulator -> select a track -> click "Autonomous Mode".
"""

import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
app = Flask(__name__)

maxSpeed = 15   # keep low initially, increase once model is accurate


def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array([image])

    steering = float(model.predict(image)[0][0])
    throttle = 1.0 - speed / maxSpeed

    print(f'Steering: {steering:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('model.h5', compile=False)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)