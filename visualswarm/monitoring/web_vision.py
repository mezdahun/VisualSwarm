"""Methods to stream vision of robot via mjpg web server"""
import io
import logging
import socketserver
from http import server
frame = None
from PIL import Image
import logging
from visualswarm.contrib import logparams
import cv2

logger = logging.getLogger('visualswarm.app')
bcolors = logparams.BColors

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
                    item = self.server.queue.get()
                    if item is not None:
                        frame = item[0]
                    jpg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype('uint8'))
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
        self.queue = None

def web_vision_process(vision_queue, port=8000):
    """Sending vision queue to simple mjpg server, on selected port"""
    address = ('', port)
    server = StreamingServer(address, StreamingHandler)
    server.queue = vision_queue
    server.serve_forever()

def start_webcam_only(port=8000):
    """In case visualswarm is installed on a raspberry pi used for observation camera, running this function
    will start a camera and publishes on a simple webserver that can be fetched form the same network."""
    from multiprocessing import Queue
    import logging
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    from picamera.exc import PiCameraValueError
    from visualswarm.contrib import webcamera, logparams
    import threading

    raw_vision_stream = Queue()
    logger = logging.getLogger('visualswarm.app')
    bcolors = logparams.BColors

    logger.info('Starting observation process (webcam)...')

    picam = PiCamera()
    picam.resolution = webcamera.RESOLUTION
    picam.framerate = webcamera.FRAMERATE
    logger.debug(f'\n{bcolors.OKBLUE}--WebCamera Params--{bcolors.ENDC}\n'
                 f'{bcolors.OKBLUE}Resolution:{bcolors.ENDC} {webcamera.RESOLUTION} px\n'
                 f'{bcolors.OKBLUE}Frame Rate:{bcolors.ENDC} {webcamera.FRAMERATE} fps')

    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(picam, size=webcamera.RESOLUTION)

    address = ('', port)
    server = StreamingServer(address, StreamingHandler)
    server.queue = raw_vision_stream

    # Starting server on different thread
    t = threading.Thread(target=server.serve_forever)
    t.start()

    # Wait a certain number of seconds to allow the camera time to warmup
    frame_id = 0
    try:
        try:
            try:
                for frame in picam.capture_continuous(raw_capture,
                                                      format=webcamera.CAPTURE_FORMAT,
                                                      use_video_port=webcamera.USE_VIDEO_PORT):
                    img = frame.array
                    raw_capture.truncate(0)
                    if raw_vision_stream.qsize() > 20:
                        raw_vision_stream.get()
                    raw_vision_stream.put((img, None, frame_id, None))
                    print(raw_vision_stream.qsize())

            except KeyboardInterrupt:
                try:
                    print("Shutting down server!")
                    t.join()
                    print("ByeBye!")
                except PiCameraValueError:
                    print("Shutting down server!")
                    t.join()
                    print("ByeBye!")
        except PiCameraValueError:
            print("Shutting down server!")
            t.join()
            print("ByeBye!")
    except KeyboardInterrupt:
        print("Shutting down server!")
        t.join()
        print("ByeBye!")
