# Video and Audio Emotion Detection

## Description
A real-time emotion detection system that analyzes both video and audio streams from uploaded video files using computer vision and deep learning techniques.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## Features
* Simultaneous video and audio emotion detection
* Real-time processing and display
* Support for multiple video formats (mp4, mov, avi)
* Automatic temporary file cleanup
* Thread-based parallel processing

## Installation
```bash
# Clone the repository
git clone [your-repository-url]
cd [repository-name]

# Install required packages
pip install -r requirements.txt
```

## Dependencies
* streamlit
* opencv-python
* fer
* numpy
* librosa
* pygame
* tensorflow
* moviepy

## Setup
1. Install all required dependencies:
```bash
pip install streamlit opencv-python fer numpy librosa pygame tensorflow moviepy
```

2. Update the audio model path in main():
```python
audio_model_path = "path/to/your/model.h5"
```

3. Ensure you have the audio emotion detection model in place

## Usage
1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface:
   * Open your browser
   * Navigate to the provided localhost URL
   * Upload a video file
   * Click "Process Video"

## Technical Details

### Emotion Detection Process
1. Video Processing:
   * Uses FER for facial emotion detection
   * Processes frames in real-time
   * Displays emotion labels on video

2. Audio Processing:
   * Extracts audio from video
   * Processes using MFCC features
   * Uses deep learning for classification

### Supported Emotions
* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

### Implementation Details
* Threading for parallel processing
* Real-time synchronization
* Automatic resource management
* Temporary file handling

## Troubleshooting

### Common Issues

1. Model Loading Errors
   * Solution: Verify model path and file existence

2. Video Processing Issues
   * Solution: Check video format compatibility
   * Solution: Ensure adequate system resources

3. Audio Processing Problems
   * Solution: Verify audio stream in video
   * Solution: Check pygame initialization

### Requirements
* Python 3.7+
* Sufficient RAM for video processing
* GPU recommended for better performance
