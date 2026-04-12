import pyrealsense2 as rs
import numpy as np
import cv2
import time

ctx = rs.context()
for dev in ctx.query_devices():
    dev.hardware_reset()
time.sleep(5)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
pipeline.start(config)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dataset_video.mp4', fourcc, 15, (1280, 720))

print("Recording... Press 'q' to stop.")
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        out.write(img)
        cv2.imshow('Recording', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
    print("Saved to dataset_video.mp4")