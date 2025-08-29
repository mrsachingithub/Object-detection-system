from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load image
img_path = "D:/OBJECT-DETECTION/chapter1/image/Parking_2.jpg"
results = model(img_path)

# Read image using OpenCV
img = cv2.imread(img_path)

# Loop through detection results
for result in results:
    boxes = result.boxes

    for box in boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get confidence score
        conf = float(box.conf[0])
        label = f"{conf:.2f}"  # Only show confidence

        # Draw thinner bounding box and confidence score
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Thickness = 1
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)  # Smaller font and thinner text

# Optional: Resize image for better visibility
img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# Display result
cv2.imshow("YOLOv8 Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()