import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videoplayback.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (1240, 700))

    # Run detection
    results = model(frame)
    annotated = results[0].plot()

    # Show resized + annotated frame
    cv2.imshow("YOLOv8 Detection - Resized", annotated)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()