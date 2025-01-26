# Whisper Transcription and Diarization

This project combines OpenAI's Whisper (via faster-whisper) and Pyannote.audio for audio transcription with speaker diarization.

## Features

- Audio transcription using faster-whisper (medium model)
- Speaker diarization using pyannote.audio
- Optimized for Apple Silicon (M1/M2)
- Outputs both raw transcription and speaker-diarized results

## Requirements

- Python 3.10+
- PyTorch
- faster-whisper
- pyannote.audio
- torchaudio
- HuggingFace account with accepted model terms

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv_310
source venv_310/bin/activate
```

2. Install dependencies:
```bash
pip install pyannote.audio faster-whisper torch torchaudio
```

3. Get a HuggingFace token and accept the user conditions at:
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/speaker-diarization-3.1

4. Set your HuggingFace token as an environment variable:
```bash
export HF_TOKEN='your_token_here'
```

## Usage

1. Place your audio file in the project directory
2. Update the input file name in `transcribe.py`
3. Run the script:
```bash
python3 transcribe.py
```

The script will generate two files:
- `transcription_raw.txt`: Raw transcription with timestamps
- `transcription_diarized.txt`: Full transcription with speaker labels

## Performance

- Uses optimal thread count for CPU cores
- Includes Voice Activity Detection (VAD)
- Groups segments from same speaker if they're ≤ 2 seconds apart 