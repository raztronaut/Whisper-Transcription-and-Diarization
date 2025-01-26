from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
import torch
import multiprocessing
import torchaudio
import time
import re
import os

# Get optimal number of CPU cores (M2 Pro has 10 CPU cores)
cpu_count = multiprocessing.cpu_count()
optimal_threads = min(cpu_count, 10)  # M2 Pro specific
optimal_workers = max(1, optimal_threads // 2)

print(f"Initializing with {optimal_threads} CPU threads and {optimal_workers} workers...")

# Get HuggingFace token from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    print("Error: HF_TOKEN environment variable not set")
    print("Please set your HuggingFace token:")
    print("export HF_TOKEN='your_token_here'")
    exit(1)

# Initialize the pipeline with your HuggingFace token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token
)

# Initialize the Whisper model with optimal settings for M2 Pro
print("Loading Whisper model...")
model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8",
    cpu_threads=optimal_threads,
    num_workers=optimal_workers
)

def process_audio(audio_file, group_segments=True):
    time_start = time.time()
    
    # Step 1: Transcribe entire audio first
    print("Starting transcription...")
    segments, info = model.transcribe(
        audio_file,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=1000),
        word_timestamps=True,
        initial_prompt="City Council Meeting",
    )
    
    # Save raw transcription
    print("Saving raw transcription...")
    with open("transcription_raw.txt", "w") as f:
        for segment in segments:
            f.write(f"{segment.start:.1f}s - {segment.end:.1f}s: {segment.text}\n")
    
    # Convert segments to list and process
    segments = list(segments)
    segments = [
        {
            "avg_logprob": s.avg_logprob,
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text,
            "words": [
                {
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                    "probability": w.probability,
                }
                for w in s.words
            ],
        }
        for s in segments
    ]
    
    time_transcribing_end = time.time()
    print(f"Finished transcription in {time_transcribing_end - time_start:.2f} seconds")
    
    # Step 2: Perform diarization
    print("Starting speaker diarization...")
    waveform, sample_rate = torchaudio.load(audio_file)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    time_diarization_end = time.time()
    print(f"Finished diarization in {time_diarization_end - time_transcribing_end:.2f} seconds")
    
    # Step 3: Merge transcription with speaker information
    print("Merging transcription with speaker information...")
    margin = 0.1  # 100ms margin for better word-speaker matching
    final_segments = []
    diarization_list = list(diarization.itertracks(yield_label=True))
    
    # Get number of unique speakers
    unique_speakers = {speaker for _, _, speaker in diarization.itertracks(yield_label=True)}
    num_speakers = len(unique_speakers)
    
    speaker_idx = 0
    n_speakers = len(diarization_list)
    
    # Process each segment
    for segment in segments:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = []
        segment_words = []
        
        # Process each word in the segment
        for word in segment["words"]:
            word_start = word["start"] - margin
            word_end = word["end"] + margin
            
            while speaker_idx < n_speakers:
                turn, _, speaker = diarization_list[speaker_idx]
                if turn.start <= word_end and turn.end >= word_start:
                    segment_text.append(word["word"])
                    word["word"] = word["word"].strip()
                    segment_words.append(word)
                    if turn.end <= word_end:
                        speaker_idx += 1
                    break
                elif turn.end < word_start:
                    speaker_idx += 1
                else:
                    break
        
        if segment_text:
            combined_text = "".join(segment_text)
            cleaned_text = re.sub(" +", " ", combined_text).strip()
            
            new_segment = {
                "avg_logprob": segment["avg_logprob"],
                "start": segment_start,
                "end": segment_end,
                "speaker": speaker,
                "text": cleaned_text,
                "words": segment_words,
            }
            final_segments.append(new_segment)
    
    # Step 4: Group segments from same speaker if they're close together
    if group_segments:
        print("Grouping segments...")
        output = []
        if not final_segments:
            return []
            
        current_group = {
            "start": final_segments[0]["start"],
            "end": final_segments[0]["end"],
            "speaker": final_segments[0]["speaker"],
            "text": final_segments[0]["text"],
            "words": final_segments[0]["words"],
            "avg_logprob": final_segments[0]["avg_logprob"],
        }
        
        for i in range(1, len(final_segments)):
            time_gap = final_segments[i]["start"] - final_segments[i-1]["end"]
            
            if (final_segments[i]["speaker"] == final_segments[i-1]["speaker"] 
                and time_gap <= 2):  # Group segments 2 seconds or less apart
                current_group["end"] = final_segments[i]["end"]
                current_group["text"] += " " + final_segments[i]["text"]
                current_group["words"].extend(final_segments[i]["words"])
            else:
                output.append(current_group)
                current_group = {
                    "start": final_segments[i]["start"],
                    "end": final_segments[i]["end"],
                    "speaker": final_segments[i]["speaker"],
                    "text": final_segments[i]["text"],
                    "words": final_segments[i]["words"],
                    "avg_logprob": final_segments[i]["avg_logprob"],
                }
        
        output.append(current_group)
        final_segments = output
    
    time_end = time.time()
    print(f"\nTotal processing time: {time_end - time_start:.2f} seconds")
    print(f"Found {num_speakers} speakers in the audio")
    
    return final_segments, num_speakers

# Process the audio file
print("\nProcessing audio file...")
segments, num_speakers = process_audio("short_test.wav")

# Save diarized output to file
print("\nSaving diarized output...")
with open("transcription_diarized.txt", "w") as f:
    f.write(f"Number of speakers detected: {num_speakers}\n")
    f.write("=====================\n\n")
    for segment in segments:
        f.write(f"[{segment['speaker']}] {segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}\n")

# Print the results
print("\nTranscription Results:")
print(f"Number of speakers detected: {num_speakers}")
print("=====================\n")
for segment in segments:
    print(f"[{segment['speaker']}] {segment['start']:.1f}s - {segment['end']:.1f}s: {segment['text']}") 