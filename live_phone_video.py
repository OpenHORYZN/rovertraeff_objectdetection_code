#!/usr/bin/env python3
"""Display live video from phone's IP Webcam."""

import cv2

url = "http://192.168.178.151:8080/video"

print(f"Connecting to: {url}")
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Failed to open camera!")
    exit(1)

print("✓ Connected! Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    # Display
    cv2.imshow('Phone Live Feed', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
