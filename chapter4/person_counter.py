"""import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os
import time

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
cap = cv2.VideoCapture(os.path.join(current_dir, "people.mp4"))

# Use the smallest model for fastest CPU detection
model = YOLO(os.path.join(current_dir, "../Yolo-Weights/yolov8n.pt"))

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load first frame
success, img = cap.read()
if not success:
    print("Could not read video")
    exit()

# Load mask
mask_path = os.path.join(current_dir, "mask.png")
mask = cv2.imread(mask_path)
if mask is None:
    print("Could not load mask image")
    exit()

mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]
totalCountUp = []
totalCountDown = []

# Performance settings
frame_skip = 1
prev_frame_time = 0

print("Starting CPU detection... Press 'q' to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    # Reduce resolution for faster CPU processing
    img = cv2.resize(img, (640, 480))
    mask_resized = cv2.resize(mask, (640, 480))
    
    imgRegion = cv2.bitwise_and(img, mask_resized)

    # Use smaller inference size for CPU
    results = model(imgRegion, stream=True, imgsz=320, verbose=False)
    
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    # Draw lines
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 3)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 3)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        
        cvzone.cornerRect(img, (x1, y1, w, h), l=6, rt=1, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(25, y1)),
                           scale=1, thickness=2, offset=5)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        # Check crossing
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 10 < cy < limitsUp[1] + 10:
            if id not in totalCountUp:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 3)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 10 < cy < limitsDown[1] + 10:
            if id not in totalCountDown:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 3)

    # Display info
    fps = 1 / (time.time() - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = time.time()
    
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Up: {len(totalCountUp)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, f"Down: {len(totalCountDown)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Fast Person Counter (CPU)", img)
    
    # Use minimal delay
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Detection completed!")
print(f"Total people going UP: {len(totalCountUp)}")
print(f"Total people going DOWN: {len(totalCountDown)}")
"""









import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os
import time

# Get the current directory (chapter4)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Use correct paths based on your file structure
cap = cv2.VideoCapture(os.path.join(current_dir, "people.mp4"))

model = YOLO(os.path.join(current_dir, "../Yolo-Weights/yolov8l.pt"))

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# --- Load first frame just to get video size ---
success, img = cap.read()
if not success:
    print("Could not read video. Check if people.mp4 exists in the chapter4 directory.")
    print(f"Looking for video at: {os.path.join(current_dir, 'people.mp4')}")
    exit()

# --- Load and resize mask to match video frame size ---
mask_path = os.path.join(current_dir, "mask.png")
mask = cv2.imread(mask_path)
if mask is None:
    print(f"Could not load mask image. Check if mask.png exists at: {mask_path}")
    exit()
    
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# Reset video back to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limitsUp = [103, 161, 296, 161]
limitsDown = [527, 489, 735, 489]

totalCountUp = []
totalCountDown = []

# For performance measurement
prev_frame_time = 0
new_frame_time = 0
frame_count = 0
start_time = time.time()

# Set smaller resolution for faster processing (optional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    success, img = cap.read()
    if not success:
        print("End of video or could not read frame")
        break
    
    frame_count += 1
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time

    imgRegion = cv2.bitwise_and(img, mask)
    
    # Start timing for this frame
    frame_start_time = time.time()
    
    # Use smaller inference size for faster processing
    results = model(imgRegion, stream=True, imgsz=640)  # Reduced from default 640
    
    detections = np.empty((0, 5))
    detection_info = []  # Store detection info for display

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                # Store detection info for display
                detection_info.append(f"[{x1:6.1f} {y1:6.1f} {x2:6.1f} {y2:6.1f} {cls:2d}]")

    # Calculate inference time
    inference_time = (time.time() - frame_start_time) * 1000  # Convert to ms
    
    resultsTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)

    # Display information on the frame (like in the screenshot)
    h, w = img.shape[:2]
    
    # Create a semi-transparent background for text
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (400, 150 + len(detection_info)*20), (0, 0, 0), -1)
    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Display detection information
    cv2.putText(img, f"Size: {w}x{h} {len(detections)} persons, {inference_time:.1f}ms", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"FPS: {fps:.1f}, Total Up: {len(totalCountUp)}, Down: {len(totalCountDown)}", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display detection coordinates
    for i, info in enumerate(detection_info):
        cv2.putText(img, info, (20, 80 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display counts in the designated areas
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("Person Counter", img)
    
    # Also print info to console (like in your screenshot)
    if frame_count % 10 == 0:  # Print every 10 frames to avoid spamming
        print(f"Size: {w}x{h} {len(detections)} persons, {inference_time:.1f}ms")
        print(f"Speed: {fps:.1f} FPS")
        for info in detection_info:
            print(info)
        print("---")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print overall performance
end_time = time.time()
total_time = end_time - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"\n=== PERFORMANCE SUMMARY ===")
print(f"Total frames processed: {frame_count}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Total people counted (Up): {len(totalCountUp)}")
print(f"Total people counted (Down): {len(totalCountDown)}")

cap.release()
cv2.destroyAllWindows()
