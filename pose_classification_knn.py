#!/usr/bin/env python3
"""
Body Posture Classification using KNN and MediaPipe Pose Landmarks
Classifies: Sitting, Standing, Laying
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import seaborn as sns
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings("ignore")


class PoseLandmarkExtractor:
    """Extract pose landmarks from frames using MediaPipe"""

    def __init__(self):
        """Initialize MediaPipe pose detection"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        )

        # MediaPipe pose has 33 landmarks, each with x, y, z coordinates
        self.num_landmarks = 33
        self.features_per_landmark = 3  # x, y, z
        self.total_features = (
            self.num_landmarks * self.features_per_landmark
        )  # 99 features

    def extract_landmarks_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract pose landmarks from a single frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            Array of 99 features (33 landmarks × 3 coordinates)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = self.pose.process(rgb_frame)

        # Initialize feature array
        features = np.zeros(self.total_features)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract x, y, z coordinates for each landmark
            for i, landmark in enumerate(landmarks):
                features[i * 3] = landmark.x  # x coordinate
                features[i * 3 + 1] = landmark.y  # y coordinate
                features[i * 3 + 2] = landmark.z  # z coordinate

        return features

    def extract_landmarks_from_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract landmarks from multiple frames

        Args:
            frames: List of frames

        Returns:
            Array of shape (num_frames, 99) containing landmark features
        """
        features_list = []

        for i, frame in enumerate(frames):
            print(f"   Extracting landmarks from frame {i+1}/{len(frames)}...")
            features = self.extract_landmarks_from_frame(frame)
            features_list.append(features)

        return np.array(features_list)

    def close(self):
        """Close the pose detector"""
        if self.pose:
            self.pose.close()


class BodyPostureClassifier:
    """KNN classifier for body posture classification"""

    def __init__(self, n_neighbors=5):
        """
        Initialize the classifier

        Args:
            n_neighbors: Number of neighbors for KNN
        """
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.class_names = ["sitting", "standing", "laying"]

    def prepare_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset from video files in Data directory

        Args:
            data_dir: Path to Data directory

        Returns:
            Tuple of (features, labels)
        """
        print("📊 Preparing dataset from video files...")

        features_list = []
        labels_list = []

        # Initialize landmark extractor
        extractor = PoseLandmarkExtractor()

        try:
            # Process each posture class
            for class_idx, posture in enumerate(["Sitting", "Standing", "laying"]):
                print(f"\n🎯 Processing {posture} posture...")

                posture_dir = Path(data_dir) / posture
                if not posture_dir.exists():
                    print(f"   ⚠️  Directory {posture} not found, skipping...")
                    continue

                # Get video files
                video_files = list(posture_dir.glob("*.mp4"))
                print(f"   📹 Found {len(video_files)} video files")

                # Process each video (limit for demonstration)
                max_videos_per_class = 10  # Limit to avoid long processing
                for i, video_path in enumerate(video_files[:max_videos_per_class]):
                    print(
                        f"   Processing video {i+1}/{min(len(video_files), max_videos_per_class)}: {video_path.name}"
                    )

                    # Extract frames
                    frames = self.extract_frames_from_video(
                        str(video_path), sample_rate=60
                    )

                    if frames:
                        # Extract landmarks from frames
                        frame_features = extractor.extract_landmarks_from_frames(frames)

                        # Add features and labels
                        features_list.extend(frame_features)
                        labels_list.extend([class_idx] * len(frame_features))

                        print(f"     ✅ Extracted {len(frame_features)} frames")
                    else:
                        print(f"     ❌ No frames extracted from {video_path.name}")

        finally:
            extractor.close()

        # Convert to numpy arrays
        features = np.array(features_list)
        labels = np.array(labels_list)

        print(f"\n📊 Dataset prepared:")
        print(f"   Total samples: {len(features)}")
        print(f"   Features per sample: {features.shape[1]}")
        print(f"   Classes: {self.class_names}")

        # Print class distribution
        unique, counts = np.unique(labels, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"   {self.class_names[class_idx]}: {count} samples")

        return features, labels

    def extract_frames_from_video(
        self, video_path: str, sample_rate: int = 60
    ) -> List[np.ndarray]:
        """Extract frames from video file"""
        frames_array = []
        video_capture = cv2.VideoCapture(video_path)

        if not video_capture.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return frames_array

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if fps <= 0:
            print(f"Warning: Unable to obtain framerate for video: {video_path}")
            video_capture.release()
            return frames_array

        frame_index = 0
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            if frame_index % sample_rate == 0:
                frames_array.append(frame)

            frame_index += 1

        video_capture.release()
        return frames_array

    def train(self, features: np.ndarray, labels: np.ndarray, test_size: float = 0.2):
        """
        Train the KNN classifier

        Args:
            features: Feature array (num_samples, 99)
            labels: Label array (num_samples,)
            test_size: Fraction of data to use for testing
        """
        print("\n🎯 Training KNN Classifier...")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )

        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train KNN classifier
        self.knn.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = self.knn.predict(X_test_scaled)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"   ✅ Training completed!")
        print(f"   📊 Test accuracy: {accuracy:.3f}")

        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.y_pred = y_pred

        self.is_trained = True

        return accuracy

    def evaluate(self):
        """Evaluate the trained classifier"""
        if not self.is_trained:
            print("❌ Classifier not trained yet!")
            return

        print("\n📊 Model Evaluation:")
        print("=" * 50)

        # Classification report
        print("\n📋 Classification Report:")
        print(
            classification_report(
                self.y_test, self.y_pred, target_names=self.class_names
            )
        )

        # Confusion matrix
        print("\n🎯 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        self.plot_confusion_matrix(cm)

        # Cross-validation score
        print("\n🔄 Cross-Validation Score:")
        cv_scores = cross_val_score(self.knn, self.X_test, self.y_test, cv=5)
        print(
            f"   Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})"
        )

    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.show()

    def predict_single_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Predict posture for a single frame

        Args:
            frame: Input frame

        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.is_trained:
            print("❌ Classifier not trained yet!")
            return None, 0.0

        # Extract landmarks
        extractor = PoseLandmarkExtractor()
        try:
            features = extractor.extract_landmarks_from_frame(frame)
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Predict
            prediction = self.knn.predict(features_scaled)[0]
            confidence = self.knn.predict_proba(features_scaled)[0].max()

            return self.class_names[prediction], confidence
        finally:
            extractor.close()

    def save_model(self, filepath: str):
        """Save the trained model"""
        import joblib

        model_data = {
            "knn": self.knn,
            "scaler": self.scaler,
            "class_names": self.class_names,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load a trained model"""
        import joblib

        model_data = joblib.load(filepath)
        self.knn = model_data["knn"]
        self.scaler = model_data["scaler"]
        self.class_names = model_data["class_names"]
        self.is_trained = model_data["is_trained"]

        print(f"✅ Model loaded from {filepath}")


def main():
    """Main function to demonstrate the complete pipeline"""
    print("🚀 Body Posture Classification Pipeline")
    print("=" * 50)

    # Initialize classifier
    classifier = BodyPostureClassifier(n_neighbors=5)

    # Prepare dataset
    features, labels = classifier.prepare_dataset("Data")

    if len(features) == 0:
        print("❌ No data extracted. Please check your Data directory.")
        return

    # Train classifier
    accuracy = classifier.train(features, labels, test_size=0.2)

    # Evaluate model
    classifier.evaluate()

    # Save model
    classifier.save_model("body_posture_classifier.pkl")

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
