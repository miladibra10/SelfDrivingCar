import argparse
import os
import shutil
import eventlet.wsgi
import socketio
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
        self.image_save = True

    def send(self, angle, throttle):
        socket.emit("steer", data={'steering_angle': str(angle), 'throttle': str(throttle)}, skip_sid=True)


@socket.on('telemetry')
def telemetry(sid, data):
    print("Data Sent to Driver Controller from Simulator")


@socket.on('connect')
def connect(sid, environ):
    print("Connection with Simulator Established")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver Controller')
    parser.add_argument('model', type=str, nargs='?', default='', help='Path for learned keras h5 model')
    parser.add_argument('image_folder', type=str, nargs='?', default='', help='Images for this run will be saved in this path')
    args = parser.parse_args()

    # TODO loading model

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING...")

    app = socketio.Middleware(socket, app)

    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
