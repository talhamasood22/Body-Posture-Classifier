import cv2
import os
import logging
from datetime import datetime


def setup_logging():

    if not os.path.exists("logs"):
        os.makedirs("logs")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/frame_extraction_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),  # Also print to console
        ],
    )

    logging.info(f"Starting frame extraction process. Log file: {log_filename}")
    return log_filename


def extract_frames_to_array(video_path):
    frames_array = []
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return frames_array

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if fps <= 0:
        logging.warning(f"Unable to obtain framerate for video: {video_path}")
        video_capture.release()
        return frames_array

    logging.info(f"Processing video: {os.path.basename(video_path)}")
    logging.info(f"  - FPS: {fps:.2f}")
    logging.info(f"  - Total frames: {total_frames}")
    logging.info(f"  - Duration: {duration:.2f} seconds")

    frame_index = 0
    extracted_count = 0

    while True:
        success, frame = video_capture.read()

        if not success:
            break

        if frame_index % 60 == 0:  # Saves 1 frame after every 60 frames
            frames_array.append(frame)

        frame_index += 1

    video_capture.release()

    logging.info(
        f"✅ Successfully extracted {len(frames_array)} frames from {os.path.basename(video_path)}"
    )
    return frames_array


def main():

    log_filename = setup_logging()

    logging.info("=" * 60)
    logging.info("BODY POSTURE CLASSIFIER - FRAME EXTRACTION")
    logging.info("=" * 60)

    laying_frames = []
    standing_frames = []
    sitting_frames = []

    logging.info("\n📁 Processing LAYING videos...")
    laying_dir = "Data/laying"
    if os.path.exists(laying_dir):
        laying_files = [f for f in os.listdir(laying_dir) if f.endswith(".mp4")]
        logging.info(f"Found {len(laying_files)} laying video files")

        for i, file_name in enumerate(laying_files, 1):
            logging.info(f"\n--- Processing laying video {i}/{len(laying_files)} ---")
            file_path = os.path.join(laying_dir, file_name)
            frames = extract_frames_to_array(file_path)
            laying_frames.append(frames)
    else:
        logging.error(f"Directory not found: {laying_dir}")

    logging.info("\n📁 Processing SITTING videos...")
    sitting_dir = "Data/Sitting"
    if os.path.exists(sitting_dir):
        sitting_files = [f for f in os.listdir(sitting_dir) if f.endswith(".mp4")]
        logging.info(f"Found {len(sitting_files)} sitting video files")

        for i, file_name in enumerate(sitting_files, 1):
            logging.info(f"\n--- Processing sitting video {i}/{len(sitting_files)} ---")
            file_path = os.path.join(sitting_dir, file_name)
            frames = extract_frames_to_array(file_path)
            sitting_frames.append(frames)
    else:
        logging.error(f"Directory not found: {sitting_dir}")

    logging.info("\n📁 Processing STANDING videos...")
    standing_dir = "Data/Standing"
    if os.path.exists(standing_dir):
        standing_files = [f for f in os.listdir(standing_dir) if f.endswith(".mp4")]
        logging.info(f"Found {len(standing_files)} standing video files")

        for i, file_name in enumerate(standing_files, 1):
            logging.info(
                f"\n--- Processing standing video {i}/{len(standing_files)} ---"
            )
            file_path = os.path.join(standing_dir, file_name)
            frames = extract_frames_to_array(file_path)
            standing_frames.append(frames)
    else:
        logging.error(f"Directory not found: {standing_dir}")

    logging.info("\n" + "=" * 60)
    logging.info("EXTRACTION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Laying videos processed: {len(laying_frames)}")
    logging.info(f"Sitting videos processed: {len(sitting_frames)}")
    logging.info(f"Standing videos processed: {len(standing_frames)}")

    total_laying_frames = sum(len(frames) for frames in laying_frames)
    total_sitting_frames = sum(len(frames) for frames in sitting_frames)
    total_standing_frames = sum(len(frames) for frames in standing_frames)

    logging.info(f"Total laying frames: {total_laying_frames}")
    logging.info(f"Total sitting frames: {total_sitting_frames}")
    logging.info(f"Total standing frames: {total_standing_frames}")
    logging.info(
        f"Total frames extracted: {total_laying_frames + total_sitting_frames + total_standing_frames}"
    )

    logging.info(f"\n✅ Frame extraction completed! Check log file: {log_filename}")

    return laying_frames, sitting_frames, standing_frames


if __name__ == "__main__":
    laying_frames, sitting_frames, standing_frames = main()
