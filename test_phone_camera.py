#!/usr/bin/env python3
"""Quick test - can we read from phone's IP Webcam?"""

import cv2
import sys

# Test URL from your config
url = "http://192.168.178.151:8080/video"

print(f"Testing connection to: {url}")
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ Failed to open camera!")
    print("Possible issues:")
    print("  1. Phone IP is wrong (check IP Webcam app)")
    print("  2. Phone not on same WiFi network")
    print("  3. Firewall blocking port 8080")
    print("  4. IP Webcam app not running")
    sys.exit(1)

print("✓ Camera opened!")

# Try to read a frame
ret, frame = cap.read()
if ret:
    print(f"✓ Got frame: {frame.shape}")
    cv2.imwrite("/tmp/test_frame.png", frame)
    print("✓ Saved to /tmp/test_frame.png")
else:
    print("❌ Failed to read frame")

cap.release()
