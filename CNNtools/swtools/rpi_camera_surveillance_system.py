# Web streaming example
# Source code from the official PiCamera package
# http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming
import threading

from visualswarm.contrib import camera
import io
import picamera
import logging
import socketserver
from threading import Condition
from http import server

PAGE = """\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""

frame = None

from PIL import Image
import threading
import time


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        global frame
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    print(self.server.queue)
                    jpg = Image.fromarray(frame.astype('uint8'))
                    print(jpg)
                    buf = io.BytesIO()
                    jpg.save(buf, format='JPEG')
                    frame_n = buf.getvalue()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_n))
                    self.end_headers()
                    self.wfile.write(frame_n)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, x, y):
        super(StreamingServer, self).__init__(x, y)
        self.queue = "test"


from picamera import PiCamera
from picamera.array import PiRGBArray
from picamera.exc import PiCameraValueError
import cv2

picam = PiCamera()
picam.resolution = camera.RESOLUTION
picam.framerate = camera.FRAMERATE
picam.sensor_mode = 4

# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(picam, size=camera.RESOLUTION)

# Wait a certain number of seconds to allow the camera time to warmup
frame_id = 0

address = ('', 8000)
server = StreamingServer(address, StreamingHandler)
threading.Thread(target=server.serve_forever).start()

for frame_raw in picam.capture_continuous(raw_capture,
                                          format=camera.CAPTURE_FORMAT,
                                          use_video_port=camera.USE_VIDEO_PORT):
    print('cap')
    frame = frame_raw.array
    print(frame.shape)
    # Clear the raw capture stream in preparation for the next frame
    raw_capture.truncate(0)

# with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
#     output = StreamingOutput()
#     # Uncomment the next line to change your Pi's Camera rotation (in degrees)
#     # camera.rotation = 90
#     camera.start_recording(output, format='mjpeg')
#     try:
#         address = ('', 8000)
#         server = StreamingServer(address, StreamingHandler)
#         server.serve_forever()
#     finally:
#         camera.stop_recording()