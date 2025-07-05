# Body Posture Classifier

This project is designed to classify body postures from video data. It extracts frames from video files and processes them for posture classification.

## Project Structure

```
Body-Posture-Classifier/
├── Data/
│   ├── laying/          # Video files for laying posture
│   ├── Sitting/         # Video files for sitting posture
│   └── Standing/        # Video files for standing posture
├── main.py              # Main script for frame extraction
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup Instructions

1. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation:**
   ```bash
   python -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

## Usage

The current `main.py` script:
- Extracts frames from video files in the Data directory
- Samples 1 frame every 60 frames (approximately 1 frame per second)
- Organizes frames by posture type (laying, sitting, standing)

To run the script:
```bash
python main.py
```

## Data Format

The project expects video files organized in the following structure:
- `Data/laying/` - Videos of people in laying position
- `Data/Sitting/` - Videos of people in sitting position  
- `Data/Standing/` - Videos of people in standing position

## Current Features

- Frame extraction from video files
- Frame sampling (1 frame per 60 frames)
- Organization by posture categories
- Error handling for video loading issues

## Next Steps

This appears to be the initial data processing stage. Future development could include:
- Frame preprocessing and feature extraction
- Machine learning model training
- Real-time posture classification
- Model evaluation and validation

## Dependencies

- **opencv-python**: Video processing and frame extraction
- **numpy**: Numerical operations
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning utilities
- **tensorflow**: Deep learning framework
- **keras**: High-level neural network API
- **Pillow**: Image processing
