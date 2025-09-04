#!/usr/bin/env python3
"""
Full Audio Transcription Script

This script transcribes a complete audio file using the best available model
for maximum accuracy. Use this for post-session transcription of recorded audio.
"""

import os
import sys
from datetime import datetime

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Missing transformers library. Install with: pip install transformers torch")

def transcribe_audio_file(audio_file_path, output_file=None):
    """
    Transcribe a complete audio file using the best available model.
    
    Args:
        audio_file_path (str): Path to the audio file
        output_file (str): Optional output file path. If None, auto-generates name.
    """
    if not TRANSFORMERS_AVAILABLE:
        return False
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    # Use the best model for full transcription
    model_id = "openai/whisper-large-v3"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    
    print(f"🎯 Using model: {model_id}")
    print(f"🖥️  Device: {device}")
    print(f"📁 Input: {audio_file_path}")
    
    try:
        # Initialize the pipeline
        print("⏳ Loading model...")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # Transcribe the audio
        print("🎤 Transcribing audio...")
        result = pipe(audio_file_path)
        transcript = result["text"]
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"transcripts/full_transcript_{base_name}_{timestamp}.txt"
        
        # Ensure transcripts directory exists
        os.makedirs("transcripts", exist_ok=True)
        
        # Save the transcript
        with open(output_file, 'w') as f:
            f.write(f"Full Audio Transcription - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Source: {audio_file_path}\n")
            f.write(f"Model: {model_id}\n")
            f.write("=" * 60 + "\n\n")
            f.write(transcript)
        
        print(f"✅ Transcription complete!")
        print(f"📄 Saved to: {output_file}")
        print(f"📊 Length: {len(transcript)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return False

def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python full_transcribe.py <audio_file> [output_file]")
        print("\nExamples:")
        print("  python full_transcribe.py lecture.wav")
        print("  python full_transcribe.py lecture.wav my_transcript.txt")
        return
    
    audio_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = transcribe_audio_file(audio_file, output_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
