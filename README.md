# Indian Sign Language (ISL) Gesture Recognition

This repository contains a real-time Indian Sign Language (ISL) gesture recognition system. It translates continuous sign language gestures into grammatically correct English and Hindi sentences, complete with Text-To-Speech (TTS) capabilities.

## Features

- **Real-Time Hand Tracking:** Uses MediaPipe (`hand_landmarker.task`) for accurate tracking of 21 3D landmarks for both hands.
- **Deep Learning Model:** An LSTM-based neural network trained on sequence data to classify up to 30 continuous gesture sequences (`gesture_lstm_model.h5`).
- **Contextual Sentence Expansion:** Intelligently maps raw gesture sequences to meaningful, context-aware English sentences (e.g., "i water" -> "I need water.").
- **Hindi Translation:** Real-time translation to Hindi using `deep-translator`.
- **Text-to-Speech (TTS):** Audio feedback for both target languages (English via Windows native TTS or `gTTS` for Hindi).
- **Buffer & Stabilization:** Custom logic ensures smooth inference, eliminating jitter and noise during real-time video processing.

## Files Structure

- `dataset_formation.py`: Script used to record and build your custom landmark dataset for gesture sequences.
- `model.py`: Script defining and training the LSTM architecture on your dataset.
- `pre.py`: The main inference script running the live webcam feed, prediction, sentence expansion, translation, and text-to-speech.
- `hand_detection.py`: Utility module dealing specifically with Mediapipe integration and hand landmark preprocessing.
- `gesture_lstm_model.h5`: The finalized compiled Keras model.
- `label_classes.npy`: NumPy array storing the list of gesture labels in the exact order the model outputs.
- `hand_landmarker.task`: Mediapipe model asset file for fast on-device inference.
- `requirements.txt`: The Python dependencies needed to run the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/umanggpt120/sign_lang.git
   cd sign_lang
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To start the real-time inference window, run:

```bash
python pre.py
```

### Controls During Inference

- `S` - Speak the currently formed sentence.
- `C` - Clear the current phrase/buffer.
- `D` - Delete the last predicted gesture from the queue.
- `H` - Switch audio output and main language to Hindi.
- `E` - Switch audio output and main language to English (default).
- `Q` - Quit the application.

## Acknowledgements

- Google MediaPipe
- TensorFlow / Keras
- OpenCV
