# Real-time object detection with natural language capabilities

## Overview
SON is a real-time speech-controlled object detection system that combines speech recognition, natural language processing, and computer vision to identify and locate objects in the physical environment.

## Features
- Speech-to-text transcription using Whisper
- Real-time object detection using YOLO
- Natural language processing with LLaMA3
- Text-to-speech feedback
- Real-world coordinate mapping
- Silence detection and audio processing

## Requirements 
python
sounddevice
numpy
opencv-python
whisper
pyttsx3
ultralytics
langchain-ollama
pydub

## Installation
1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the required models:
   - Whisper large-v3-turbo model
   - YOLO model (yolo11x.pt)
   - LLaMA 3.1 model (via Ollama)

## Usage
Run the main script:
```bash
python SON.py
```

The system will:
1. Listen for speech input
2. Transcribe the speech to text
3. Extract object names from the transcribed text
4. Activate the camera to detect mentioned objects
5. Display real-world coordinates of detected objects
6. Provide voice feedback for undetected objects

## System Components

### Speech Detection
- Continuously monitors audio input
- Uses RMS-based speech detection
- Implements silence tolerance for better speech segmentation

### Object Detection
- Real-time object detection using YOLO
- Converts pixel coordinates to real-world measurements
- Displays bounding boxes with coordinate information
- Implements timeout mechanism for search operations

### Natural Language Processing
- Extracts object names from transcribed text
- Uses LLaMA model through Langchain for text processing
- Returns clean list of detected objects

## Parameters
- Sample Rate: 16000 Hz
- Silence Threshold: 0.001
- Chunk Duration: 1 second
- Silence Tolerance: 4 seconds
- Search Timeout: 15 seconds

