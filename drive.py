import argparse
import base64
import os
import shutil
from datetime import datetime
from io import BytesIO
from keras.models import load_model
import utils
import eventlet.wsgi
import numpy
import socketio
from PIL import Image
from flask import Flask

socket = socketio.Server()
app = Flask(__name__)
model = None
image_list = None
MAX_SPEED = 30
MIN_SPEED = 10
speed_limit = MAX_SPEED


class Sender:
    def __init__(self):
        self.image_save = False

    def send(self, angle, throttle):
        socket.emit("steer", data={'steering_angle': str(angle), 'throttle': str(throttle)}, skip_sid=True)


sender = Sender()


@socket.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        if sender.image_save:
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

        try:
            image = numpy.asarray(image)
            image = utils.preprocess(image)
            image = numpy.array([image])
            steering_angle = float(model.predict(image, batch_size=1))
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2
            sender.send(steering_angle, throttle)
        except Exception as e:
            print(e)

    else:
        socket.emit('manual', data={}, skip_sid=True)


@socket.on('connect')
def connect(sid, environ):
    print("Connection with Simulator Established")
    sender.send(0, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver Controller')
    parser.add_argument('model', type=str, nargs='?', default='', help='Path for learned keras h5 model')
    parser.add_argument('image_folder', type=str, nargs='?', default='', help='Images for this run will be saved in this path')
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        sender.image_save = True
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING...")

    app = socketio.Middleware(socket, app)

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
