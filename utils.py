import cv2

# Draw traffic light on frame
def draw_traffic_light(frame, is_green, count, road_id, countdown, GREEN, RED, WHITE):
    # Draw traffic light circle
    light_color = GREEN if is_green else RED
    cv2.circle(frame, (30, 50), 20, light_color, -1)

    # Display info
    status = "GREEN" if is_green else "RED"
    cv2.putText(frame, f"Road {road_id} | {status} | Count: {count}",
                (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, light_color, 2)

    # Display countdown
    cv2.putText(frame, f"Timer: {countdown}s", (60, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    return frame


# Read and resize a frame from video
def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return cv2.resize(frame, (480, 320))

# Count relevant vehicle classes
def count_vehicles(frame, model, vehicle_classes):
    results = model(frame)[0]
    return sum(1 for c in results.boxes.cls if int(c) in vehicle_classes)
