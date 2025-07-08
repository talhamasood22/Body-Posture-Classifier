#!/usr/bin/env python3
"""
Test Body Posture Classification on a Single Video
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path


def extract_frames_from_video(video_path: str, sample_rate: int = 60):
    """Extract frames from video file"""
    frames_array = []
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return frames_array

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(
        f"Processing: {os.path.basename(video_path)} (FPS: {fps:.1f}, Duration: {duration:.1f}s)"
    )

    frame_index = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break

        if frame_index % sample_rate == 0:
            frames_array.append(frame)

        frame_index += 1

    video_capture.release()
    print(f"✅ Extracted {len(frames_array)} frames")
    return frames_array


def extract_landmarks_from_frame(frame):
    """Extract pose landmarks from a single frame"""
    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    )

    try:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = pose.process(rgb_frame)

        # Initialize feature array (33 landmarks × 3 coordinates = 99 features)
        features = np.zeros(99)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract x, y, z coordinates for each landmark
            for i, landmark in enumerate(landmarks):
                features[i * 3] = landmark.x  # x coordinate
                features[i * 3 + 1] = landmark.y  # y coordinate
                features[i * 3 + 2] = landmark.z  # z coordinate

        return features
    finally:
        pose.close()


def test_video_classification(
    video_path: str, model_path: str = "body_posture_classifier.pkl"
):
    """Test classification on a single video"""
    print("🎬 Testing Video Classification")
    print("=" * 40)

    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found!")
        print("Please run the training first or use the notebook.")
        return

    print(f"📦 Loading model from {model_path}...")
    model_data = joblib.load(model_path)
    knn = model_data["knn"]
    scaler = model_data["scaler"]
    class_names = model_data["class_names"]

    print(f"✅ Model loaded! Classes: {class_names}")

    # Extract frames
    print(f"\n📹 Processing video: {video_path}")
    frames = extract_frames_from_video(video_path, sample_rate=60)

    if not frames:
        print("❌ No frames extracted!")
        return

    # Extract landmarks and predict
    predictions = []
    confidences = []

    print(f"\n🔍 Analyzing {len(frames)} frames...")

    for i, frame in enumerate(frames):
        if i % 5 == 0:  # Progress update every 5 frames
            print(f"   Processing frame {i+1}/{len(frames)}...")

        # Extract landmarks
        features = extract_landmarks_from_frame(frame)

        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = knn.predict(features_scaled)[0]
        confidence = knn.predict_proba(features_scaled)[0].max()

        predictions.append(prediction)
        confidences.append(confidence)

    # Analyze results
    print(f"\n📊 Classification Results:")
    print(f"   Total frames analyzed: {len(frames)}")

    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"   Predictions by class:")
    for class_idx, count in zip(unique, counts):
        percentage = (count / len(predictions)) * 100
        print(f"     {class_names[class_idx]}: {count} frames ({percentage:.1f}%)")

    # Average confidence
    avg_confidence = np.mean(confidences)
    print(f"   Average confidence: {avg_confidence:.3f}")

    # Most common prediction
    most_common_idx = np.argmax(counts)
    most_common_class = class_names[unique[most_common_idx]]
    most_common_count = counts[most_common_idx]
    most_common_percentage = (most_common_count / len(predictions)) * 100

    print(f"\n🎯 Final Classification:")
    print(f"   Predicted posture: {most_common_class}")
    print(f"   Confidence: {most_common_percentage:.1f}% of frames")

    # Visualize results
    plt.figure(figsize=(12, 4))

    # Prediction distribution
    plt.subplot(1, 3, 1)
    plt.bar(
        class_names, [counts[i] if i in unique else 0 for i in range(len(class_names))]
    )
    plt.title("Frame Predictions by Class")
    plt.ylabel("Number of Frames")
    plt.xticks(rotation=45)

    # Confidence over time
    plt.subplot(1, 3, 2)
    plt.plot(confidences)
    plt.title("Confidence Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)

    # Prediction sequence
    plt.subplot(1, 3, 3)
    plt.plot(predictions, "o-", alpha=0.7)
    plt.title("Prediction Sequence")
    plt.xlabel("Frame Number")
    plt.ylabel("Class Index")
    plt.yticks(range(len(class_names)), class_names)

    plt.tight_layout()
    plt.show()

    print(f"\n✅ Video classification completed!")


def main():
    """Main function"""
    # Test with a sample video
    data_dir = Path("Data")

    # Find a sample video from each class
    sample_videos = []
    for posture in ["Sitting", "Standing", "laying"]:
        posture_dir = data_dir / posture
        if posture_dir.exists():
            video_files = list(posture_dir.glob("*.mp4"))
            if video_files:
                sample_videos.append(str(video_files[0]))

    if not sample_videos:
        print("❌ No video files found in Data directory!")
        return

    print("🎬 Available sample videos:")
    for i, video in enumerate(sample_videos):
        print(f"   {i+1}. {os.path.basename(video)}")

    # Test with the first available video
    test_video = sample_videos[0]
    print(f"\n🧪 Testing with: {os.path.basename(test_video)}")

    test_video_classification(test_video)


if __name__ == "__main__":
    main()
