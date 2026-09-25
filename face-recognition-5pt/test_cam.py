import cv2
import sys

print("Testing camera index 0...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("DirectShow failed, trying default backend...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    sys.exit(1)

ret, frame = cap.read()
if ret and frame is not None:
    print(f"SUCCESS: Frame captured successfully! Shape: {frame.shape}")
    cv2.imwrite("cam_test.jpg", frame)
    print("Saved test snapshot to cam_test.jpg")
else:
    print("ERROR: Failed to grab frame from camera.")
cap.release()
