import cv2
import time
import threading

class Reader:

    frame_buffer = []
    stop = False

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        pass

    def __call__(self):
        thread = threading.Thread(target=self.read)
        thread.start()
        pass

    def read(self):
        while not self.stop :
            success, frame = self.cap.read()  # Read a frame from the camera
            if not success:
                break
            else:
                self.frame_buffer.append(frame)  # Add the frame to the buffer
                time.sleep(0.03)  # Adjust the sleep time to control the frame processing rate
    
    def capture(self):
        ret = False
        frame = None
        if(len(self.frame_buffer) > 0):
            ret = True
            frame = self.frame_buffer[-1]
            self.frame_buffer.clear()
        return ret, frame


    def terminate(self):
        self.stop = True
        self.cap.release()
        pass

def detect(img):
    time.sleep(0.5)
    return img

source = 'rtsp://210.99.70.120:1935/live/cctv001.stream'
reader = Reader(source)
reader()
while True:
    ret, im0 = reader.capture()
    if ret:
        frame = detect(im0)
        cv2.imshow('YOLO', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break