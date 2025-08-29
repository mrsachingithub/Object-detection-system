import numpy as np
import cv2
import cvzone
import math
from ultralytics import YOLO
from sort import Sort

# === Load video and model ===
cap = cv2.VideoCapture(r"D:\OBJECT-DETECTION\chapter3\cars.mp4")

if not cap.isOpened():
    raise RuntimeError("Could not open video file. Check the path or file format.")

model = YOLO("../Yolo-weights/yolov8l.pt")

# === Load mask and graphics ===
mask = cv2.imread(r"d:/OBJECT-DETECTION/chapter3/mask.png")
imgGraphics = cv2.imread(r"d:/OBJECT-DETECTION/chapter3/graphics.png", cv2.IMREAD_UNCHANGED)

# === Resize mask to match video frame ===
success, img = cap.read()
if not success:
    raise RuntimeError("Failed to read video frame.")
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# === Initialize SORT tracker ===
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# === Define vehicle classes ===
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

vehicle_classes = {"car", "truck", "bus", "motorbike"}
limits = [400, 297, 673, 297]
totalCount = []

# === Main loop ===
while True:
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)

    if imgGraphics is not None:
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # === Run YOLO inference ===
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        # 🔹 Print YOLO’s built-in summary (like "2 cars, 26.0ms")
        print(r)

        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Only consider vehicles
            if currentClass in vehicle_classes and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

                # 🔹 Print bounding box and class in same style as screenshot
                print(f"[ {x1:.2f}  {y1:.2f}  {x2:.2f}  {y2:.2f}  {cls}]")

    # === Update tracker ===
    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("Image", img)

    # Stop button with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()

