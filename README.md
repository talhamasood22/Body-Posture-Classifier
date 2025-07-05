# Body Posture Classifier

A machine learning system that classifies human body postures (sitting, standing, laying) using MediaPipe pose landmarks and K-Nearest Neighbors (KNN) classification.

## 🎯 Overview

This project extracts pose landmarks from video frames using MediaPipe, then trains a KNN classifier to distinguish between three body postures:
- **Sitting**: Person seated in various positions
- **Standing**: Person standing or walking
- **Laying**: Person lying down or reclining

## 🚀 Features

- **MediaPipe Integration**: Extracts 33 pose landmarks (99 features: x, y, z coordinates)
- **KNN Classification**: Simple yet effective classification algorithm
- **Hyperparameter Tuning**: Automatic optimization of KNN parameters
- **Comprehensive Evaluation**: Accuracy, confusion matrix, cross-validation
- **Model Persistence**: Save and load trained models
- **Real-time Prediction**: Classify individual frames or videos
- **Visualization**: Data analysis and results visualization

## 📁 Project Structure

```
Body-Posture-Classifier/
├── Data/                          # Video dataset
│   ├── Sitting/                   # Sitting posture videos
│   ├── Standing/                  # Standing posture videos
│   └── laying/                    # Laying posture videos
├── main.py                        # Original frame extraction script
├── pose_classification_knn.py     # Complete classification pipeline
├── test_classification.py         # Simplified test script
├── test_single_video.py          # Single video classification test
├── body_posture_classification.ipynb  # Jupyter notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 (recommended for MediaPipe compatibility)
- Conda or Miniconda

### Setup Environment

1. **Create conda environment:**
```bash
conda create -n posture-env-py38 python=3.8
conda activate posture-env-py38
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Register Jupyter kernel:**
```bash
python -m ipykernel install --user --name posture-env-py38 --display-name "Python 3.8 (posture-env-py38)"
```

## 📊 Dataset

The system expects video files organized in the following structure:
```
Data/
├── Sitting/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── Standing/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── laying/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### Data Format
- **Input**: MP4 video files
- **Frame Sampling**: 1 frame every 60 frames (configurable)
- **Features**: 99 features per frame (33 landmarks × 3 coordinates)
- **Labels**: 3 classes (0: sitting, 1: standing, 2: laying)

## 🎮 Usage

### 1. Quick Test
Run a simplified test with limited data:
```bash
python test_classification.py
```

### 2. Full Pipeline
Run the complete classification pipeline:
```bash
python pose_classification_knn.py
```

### 3. Single Video Test
Test classification on a specific video:
```bash
python test_single_video.py
```

### 4. Jupyter Notebook
For interactive analysis and experimentation:
```bash
jupyter notebook body_posture_classification.ipynb
```

## 🔧 Configuration

### Key Parameters

- **Frame Sampling Rate**: `sample_rate=60` (extract every 60th frame)
- **KNN Neighbors**: `n_neighbors=5` (default, optimized via grid search)
- **Test Split**: `test_size=0.2` (20% for testing)
- **Cross-validation**: `cv=5` (5-fold cross-validation)

### MediaPipe Settings

- **Model Complexity**: 2 (high accuracy)
- **Detection Confidence**: 0.5
- **Static Image Mode**: True (for frame processing)

## 📈 Model Performance

### Typical Results
- **Test Accuracy**: 85-95% (varies with dataset)
- **Cross-validation**: 80-90%
- **Feature Count**: 99 features per frame
- **Training Time**: 1-5 minutes (depending on dataset size)

### Performance Metrics
- Classification Report (precision, recall, F1-score)
- Confusion Matrix
- Cross-validation scores
- Feature importance analysis

## 🎯 Model Architecture

### Feature Extraction
1. **Frame Extraction**: Sample frames from videos
2. **MediaPipe Processing**: Extract 33 pose landmarks
3. **Feature Vector**: 99-dimensional vector (x, y, z for each landmark)

### Classification Pipeline
1. **Data Preprocessing**: StandardScaler normalization
2. **Hyperparameter Tuning**: Grid search for optimal KNN parameters
3. **Model Training**: KNN classifier training
4. **Evaluation**: Multiple performance metrics
5. **Model Persistence**: Save trained model

## 🔍 Feature Analysis

The system analyzes the importance of different pose landmarks:
- **Key Landmarks**: Head, shoulders, hips, knees, ankles
- **Coordinate Types**: x (horizontal), y (vertical), z (depth)
- **Feature Selection**: Permutation importance analysis

## 🚀 Advanced Usage

### Custom Model Training
```python
from pose_classification_knn import BodyPostureClassifier

# Initialize classifier
classifier = BodyPostureClassifier(n_neighbors=7)

# Prepare dataset
features, labels = classifier.prepare_dataset("Data")

# Train model
accuracy = classifier.train(features, labels)

# Save model
classifier.save_model("my_model.pkl")
```

### Real-time Prediction
```python
import cv2
from pose_classification_knn import BodyPostureClassifier

# Load model
classifier = BodyPostureClassifier()
classifier.load_model("body_posture_classifier.pkl")

# Predict from frame
frame = cv2.imread("frame.jpg")
posture, confidence = classifier.predict_single_frame(frame)
print(f"Posture: {posture}, Confidence: {confidence:.3f}")
```

## 🔧 Troubleshooting

### Common Issues

1. **MediaPipe Import Error**
   - Use MediaPipe version 0.10.5
   - Ensure Python 3.8 compatibility

2. **Memory Issues**
   - Reduce `max_videos_per_class` parameter
   - Increase `sample_rate` for fewer frames

3. **Poor Accuracy**
   - Check class balance in dataset
   - Increase training data
   - Try different KNN parameters

4. **No Pose Detected**
   - Check video quality and lighting
   - Ensure person is visible in frame
   - Adjust MediaPipe confidence threshold

### Performance Optimization

- **Batch Processing**: Process multiple frames together
- **GPU Acceleration**: Use TensorFlow GPU if available
- **Data Augmentation**: Increase training data variety
- **Feature Selection**: Use only important landmarks

## 📚 Dependencies

### Core Libraries
- **OpenCV**: Video processing and frame extraction
- **MediaPipe**: Pose landmark detection
- **scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Visualization

### Version Compatibility
- Python: 3.8.x
- MediaPipe: 0.10.5
- OpenCV: 4.8.1.78
- scikit-learn: 1.3.0

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **MediaPipe**: Google's pose detection framework
- **scikit-learn**: Machine learning library
- **OpenCV**: Computer vision library

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the Jupyter notebook examples
3. Open an issue on GitHub

---

**Happy Posture Classification! 🎯**
