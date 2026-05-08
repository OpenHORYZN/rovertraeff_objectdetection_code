# sudo apt install ffmpeg libx264-dev
# or instead of avc1: 
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('dataset_video.avi', fourcc, 15, (1280, 720)) or 800?

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

fourcc = cv2.VideoWriter_fourcc(*'avc1')
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

# --------------------------------------------------------------------------------
# easiest only for color stream
#pip install ffmpeg-python pyrealsense2 numpy
#sudo apt install ffmpeg

import pyrealsense2 as rs
import numpy as np
import ffmpeg

W, H, FPS = 1280, 800, 30

# FFmpeg process — writes H.264 MP4 directly
process = (
    ffmpeg
    .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{W}x{H}', r=FPS)
    .output('dataset_video.mp4', vcodec='libx264', pix_fmt='yuv420p', crf=18)
    .overwrite_output()
    .run_async(pipe_stdin=True)
)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)  # color only, no depth
pipeline.start(config)

print("Recording... Press Ctrl+C to stop.")
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        process.stdin.write(img.tobytes())
finally:
    pipeline.stop()
    process.stdin.close()
    process.wait()
    print("Saved to dataset_video.mp4")

# ----------------------------------------------------------------------------------

import pyrealsense2 as rs
import numpy as np
import cv2

# --- RECORDING ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
config.enable_record_to_file('recording.bag')

pipeline.start(config)
input("Recording... Press Enter to stop.")
pipeline.stop()

# --- PLAYBACK (fresh objects!) ---
pipeline2 = rs.pipeline()
config2 = rs.config()
config2.enable_device_from_file('recording.bag')

pipeline2.start(config2)
try:
    while True:
        frames = pipeline2.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        cv2.imshow('Playback', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline2.stop()
    cv2.destroyAllWindows()