# Body Posture Classifier

A machine learning project for classifying body postures from video data using MediaPipe pose detection and computer vision techniques.

## Project Structure

```
Body-Posture-Classifier/
├── Data/
│   ├── laying/          # Videos of laying down postures
│   ├── Sitting/         # Videos of sitting postures  
│   └── Standing/        # Videos of standing postures
├── main.py              # Main script for frame extraction
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Setup Instructions

### 1. Create Conda Environment with Python 3.8

MediaPipe works best with Python 3.8. Create a new conda environment:

```bash
conda create -n posture-env-py38 python=3.8 -y
conda activate posture-env-py38
```

### 2. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python==4.11.0.86 numpy==1.24.3 matplotlib==3.7.5 scikit-learn==1.3.2 tensorflow==2.13.0 keras==2.13.1 pillow==10.4.0 mediapipe==0.10.5 jupyter notebook ipykernel
```

### 3. Set up Jupyter Kernel

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name posture-env-py38 --display-name "Python 3.8 (Posture Analysis)"
```

### 4. Verify Installation

Test MediaPipe installation:

```bash
python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); mp_pose = mp.solutions.pose; print('MediaPipe pose import successful!')"
```

## Usage

### Frame Extraction

Run the main script to extract frames from videos:

```bash
conda activate posture-env-py38
python main.py
```

This will:
- Process videos from the Data/ directory
- Extract one frame every 60 frames (configurable)
- Save frames to organized directories
- Log progress to `frame_extraction.log`

### Jupyter Notebook

Start Jupyter Notebook for interactive analysis:

```bash
conda activate posture-env-py38
jupyter notebook
```

Select the "Python 3.8 (Posture Analysis)" kernel when creating new notebooks.

## MediaPipe Integration

This project uses MediaPipe for pose keypoint extraction:

- **Version**: 0.10.5 (stable for Python 3.8)
- **Features**: Pose detection, landmark extraction, real-time processing
- **Compatibility**: Tested with Python 3.8 on macOS ARM64

## Data Organization

The project expects video data organized as follows:

- `Data/laying/` - Videos of people laying down
- `Data/Sitting/` - Videos of people sitting  
- `Data/Standing/` - Videos of people standing

Each video should be in MP4 format with clear posture examples.

## Development

### Environment Management

- **Primary Environment**: `posture-env-py38` (Python 3.8)
- **Backup Environment**: `posture-env` (Python 3.11 - may have MediaPipe issues)

### Key Dependencies

- **OpenCV**: Video processing and frame extraction
- **MediaPipe**: Pose detection and keypoint extraction
- **TensorFlow/Keras**: Machine learning models
- **NumPy/Matplotlib**: Data manipulation and visualization
- **scikit-learn**: Traditional ML algorithms

## Troubleshooting

### MediaPipe Import Issues

If you encounter MediaPipe import errors:

1. Ensure you're using Python 3.8
2. Use MediaPipe version 0.10.5
3. Activate the correct environment: `conda activate posture-env-py38`

### Jupyter Kernel Issues

If Jupyter can't find the kernel:

1. Reinstall the kernel: `python -m ipykernel install --user --name posture-env-py38 --display-name "Python 3.8 (Posture Analysis)"`
2. Restart Jupyter
3. Select the correct kernel when creating notebooks

## License

This project is for educational and research purposes.
