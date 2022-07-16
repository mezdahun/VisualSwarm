# # Web streaming example
# # Source code from the official PiCamera package
# # http://picamera.readthedocs.io/en/latest/recipes2.html#web-streaming
# import threading
#
# from visualswarm.contrib import camera
# import io
# import picamera
# import logging
# import socketserver
# from threading import Condition
# from http import server
#
# PAGE = """\
# <html>
# <head>
# <title>Raspberry Pi - Surveillance Camera</title>
# </head>
# <body>
# <center><h1>Raspberry Pi - Surveillance Camera</h1></center>
# <center><img src="stream.mjpg" width="640" height="480"></center>
# </body>
# </html>
# """
#
#
# # class StreamingOutput(object):
# #     def __init__(self):
# #         self.frame = None
# #         self.buffer = io.BytesIO()
# #         self.condition = Condition()
# #
# #     def write(self, buf):
# #         if buf.startswith(b'\xff\xd8'):
# #             # New frame, copy the existing buffer's content and notify all
# #             # clients it's available
# #             self.buffer.truncate()
# #             with self.condition:
# #                 self.frame = self.buffer.getvalue()
# #                 self.condition.notify_all()
# #             self.buffer.seek(0)
# #         return self.buffer.write(buf)
#
# frame = None
#
# class StreamingHandler(server.BaseHTTPRequestHandler):
#     global frame
#     def do_GET(self):
#         if self.path == '/':
#             self.send_response(301)
#             self.send_header('Location', '/index.html')
#             self.end_headers()
#         elif self.path == '/index.html':
#             content = PAGE.encode('utf-8')
#             self.send_response(200)
#             self.send_header('Content-Type', 'text/html')
#             self.send_header('Content-Length', len(content))
#             self.end_headers()
#             self.wfile.write(content)
#         elif self.path == '/stream.mjpg':
#             self.send_response(200)
#             self.send_header('Age', 0)
#             self.send_header('Cache-Control', 'no-cache, private')
#             self.send_header('Pragma', 'no-cache')
#             self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
#             self.end_headers()
#             try:
#                 while True:
#                     # with output.condition:
#                     #     output.condition.wait()
#                     #     frame = output.frame
#                     #     help(frame)
#                     self.wfile.write(b'--FRAME\r\n')
#                     self.send_header('Content-Type', 'image/jpeg')
#                     self.send_header('Content-Length', len(frame))
#                     self.end_headers()
#                     self.wfile.write(frame)
#                     self.wfile.write(b'\r\n')
#             except Exception as e:
#                 logging.warning(
#                     'Removed streaming client %s: %s',
#                     self.client_address, str(e))
#         else:
#             self.send_error(404)
#             self.end_headers()
#
#
import threading


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True
#
#
# from picamera import PiCamera
# from picamera.array import PiRGBArray
# from picamera.exc import PiCameraValueError
# import cv2
#
# picam = PiCamera()
# picam.resolution = camera.RESOLUTION
# picam.framerate = camera.FRAMERATE
# picam.sensor_mode = 4
#
# # Generates a 3D RGB array and stores it in rawCapture
# raw_capture = PiRGBArray(picam, size=camera.RESOLUTION)
#
# # Wait a certain number of seconds to allow the camera time to warmup
# frame_id = 0
#
# address = ('', 8000)
# server = StreamingServer(address, StreamingHandler)
# threading.Thread(target=server.serve_forever).start()
#
# for frame_raw in picam.capture_continuous(raw_capture,
#                                       format=camera.CAPTURE_FORMAT,
#                                       use_video_port=camera.USE_VIDEO_PORT):
#     frame = frame_raw.array.tobytes()
#     print('cap')
#     # Clear the raw capture stream in preparation for the next frame
#     raw_capture.truncate(0)
#
# # with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
# #     output = StreamingOutput()
# #     # Uncomment the next line to change your Pi's Camera rotation (in degrees)
# #     # camera.rotation = 90
# #     camera.start_recording(output, format='mjpeg')
# #     try:
# #         address = ('', 8000)
# #         server = StreamingServer(address, StreamingHandler)
# #         server.serve_forever()
# #     finally:
# #         camera.stop_recording()

import io
import socket
import struct
import time
import picamera

def start_streaming_server():
    import io
    import socket
    import struct
    from PIL import Image

    # Start a socket listening for connections on 0.0.0.0:8000 (0.0.0.0 means
    # all interfaces)
    server_socket = socket.socket()
    server_socket.bind(('0.0.0.0', 8000))
    server_socket.listen(0)

    # Accept a single connection and make a file-like object out of it
    connection = server_socket.accept()[0].makefile('rb')
    try:
        while True:
            # Read the length of the image as a 32-bit unsigned int. If the
            # length is zero, quit the loop
            image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
            if not image_len:
                break
            # Construct a stream to hold the image data and read the image
            # data from the connection
            image_stream = io.BytesIO()
            image_stream.write(connection.read(image_len))
            # Rewind the stream, open it as an image with PIL and do some
            # processing on it
            image_stream.seek(0)
            image = Image.open(image_stream)
            print('Image is %dx%d' % image.size)
            image.verify()
            print('Image is verified')
    finally:
        connection.close()
        server_socket.close()


threading.Thread(target=start_streaming_server).start()

client_socket = socket.socket()
client_socket.connect(('0.0.0.0', 8000))
connection = client_socket.makefile('wb')
try:
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.framerate = 30
        time.sleep(2)
        start = time.time()
        count = 0
        stream = io.BytesIO()
        # Use the video-port for captures...
        for foo in camera.capture_continuous(stream, 'jpeg',
                                             use_video_port=True):
            connection.write(struct.pack('<L', stream.tell()))
            connection.flush()
            stream.seek(0)
            connection.write(stream.read())
            count += 1
            if time.time() - start > 30:
                break
            stream.seek(0)
            stream.truncate()
    connection.write(struct.pack('<L', 0))
finally:
    connection.close()
    client_socket.close()
    finish = time.time()
print('Sent %d images in %d seconds at %.2ffps' % (
    count, finish-start, count / (finish-start)))
