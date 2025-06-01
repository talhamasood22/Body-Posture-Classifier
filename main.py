import cv2
import os


def extract_frames_to_array(video_path):
    frames_array = []
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return frames_array

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("unable to obtain the framerate for the video", video_path)
        video_capture.release()
        return frames_array

    frame_index = 0

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        if frame_index % 60 == 0:  # Saves 1 frame after every 60 frames
            frames_array.append(frame)

        frame_index += 1

    video_capture.release()

    print(f"Successfully extracted {len(frames_array)} frames into an array.")
    return frames_array


laying_frames = []
standing_frames = []
sitting_frames = []


for file_name in os.listdir("Dataset/laying"):
    file_path = os.path.join("Dataset/laying", file_name)
    laying_frames.append(extract_frames_to_array(file_path))

for file_name in os.listdir("Dataset/Sitting"):
    file_path = os.path.join("Dataset/Sitting", file_name)
    sitting_frames.append(extract_frames_to_array(file_path))

for file_name in os.listdir("Dataset/Standing"):
    file_path = os.path.join("Dataset/Standing", file_name)
    standing_frames.append(extract_frames_to_array(file_path))


print(laying_frames)
