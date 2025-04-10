import cv2
import time
from ultralytics import YOLO
import numpy as np
from utils import count_vehicles, draw_traffic_light, read_frame
# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Constants
DEFAULT_GREEN_TIME = 30
MAX_EXTENSION_TIME = 20
VEHICLE_THRESHOLD = 30
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Video feeds
cap1 = cv2.VideoCapture("road1.mp4")
cap2 = cv2.VideoCapture("road0.mp4")

# Colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


# Toggle for alternating roads
toggle = True

while True:
    frame1 = read_frame(cap1)
    frame2 = read_frame(cap2)

    count1 = count_vehicles(frame1, model, VEHICLE_CLASSES)
    count2 = count_vehicles(frame2, model, VEHICLE_CLASSES)
    print(f"Counts -> Road 1: {count1}, Road 2: {count2}")

    road_order = [0, 1] if toggle else [1, 0]
    toggle = not toggle

    for i in road_order:
        duration = DEFAULT_GREEN_TIME

        if i == 0:
            active = count1 > VEHICLE_THRESHOLD and count1 > count2
        else:
            active = count2 > VEHICLE_THRESHOLD and count2 > count1

        if active:
            duration += MAX_EXTENSION_TIME

        print(f"Road {i+1} GREEN for {duration}s")

        for sec in range(duration, 0, -1):  # countdown timer
            frame1_display = draw_traffic_light(read_frame(cap1), i == 0, count1, 1, sec, GREEN, RED, WHITE)
            frame2_display = draw_traffic_light(read_frame(cap2), i == 1, count2, 2, sec, GREEN, RED, WHITE)

            combined = np.hstack((frame1_display, frame2_display))
            cv2.imshow("Traffic Control Simulation", combined)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap1.release()
                cap2.release()
                cv2.destroyAllWindows()
                exit()

            time.sleep(1 / 10)  # Approx 30 FPS real-time playback
