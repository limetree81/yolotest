import cv2
from flask import Flask, Response, render_template
import threading
import time
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)
model = YOLO('D:\\jongp\\rstream\\wandlab-cv-streamer\\lab\\wandlab\\v8sbest.pt')  # Initialize YOLO object detector
camera = cv2.VideoCapture('rtsp://210.99.70.120:1935/live/cctv001.stream')  # Open the camera
frame_buffer = []  # Initialize frame buffer
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_count = 0

# visual information
annotator = None
start_time = 0
end_time = 0
# device information
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def detect_objects(frame):
    global start_time
    # Perform object detection using YOLO
    # Replace this with your actual object detection code
    # This function should return the frame with detected objects drawn on it
    start_time = time.time()
    results = predict(frame)
    im0, _ = plot_bboxes(results, frame)
    display_fps(im0)
    return im0

def predict(im0):
    results = model(im0)
    return results

def display_fps(im0):
    end_time = time.time()
    fps = 1 / np.round(end_time - start_time, 2)
    text = f'FPS: {int(fps)}'
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    gap = 10
    cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
    cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

def plot_bboxes(results, im0):
    class_ids = []
    annotator = Annotator(im0, 3, results[0].names)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()
    names = results[0].names
    for box, cls in zip(boxes, clss):
        class_ids.append(cls)
        annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
    return im0, class_ids

def process_frames():
    global frame_buffer
    while True:
        if len(frame_buffer) > 0:
            frame = frame_buffer.pop(0)  # Get the oldest frame from the buffer
            frame_buffer.clear()
            frame_with_objects = detect_objects(frame)  # Perform object detection
            cv2.imshow('frame', frame_with_objects)  # Display the frame with objects
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def capture_frames():
    global frame_buffer
    while True:
        success, frame = camera.read()  # Read a frame from the camera
        if not success:
            break
        else:
            frame_buffer.append(frame)  # Add the frame to the buffer
            time.sleep(0.03)  # Adjust the sleep time to control the frame processing rate

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Stream the frames

def gen_frames():
    while True:
        if len(frame_buffer) > 0:
            frame = frame_buffer[-1]  # Get the latest frame from the buffer
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield the frame as JPEG data

if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_frames)
    process_thread = threading.Thread(target=process_frames)
    capture_thread.start()
    process_thread.start()
    app.run()
