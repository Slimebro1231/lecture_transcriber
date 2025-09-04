#!/usr/bin/env python3
"""
Two-Pass Lecture Transcription System

This system implements:
1. Streaming Pass: Real-time transcription using a fast, smaller model
2. Refining Pass: High-accuracy transcription using a larger model
3. Resumable transcription from specific timestamps
4. Output in plain text and markdown formats for AI consumption

Usage:
    python two_pass_transcriber.py --input audio_file.wav --output transcript.txt
    python two_pass_transcriber.py --stream --input audio_file.wav
    python two_pass_transcriber.py --resume --input audio_file.wav --partial partial.txt
"""

import argparse
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import threading
import queue
import tempfile

class TwoPassTranscriber:
    def __init__(self, whisper_path: str = "./whisper.cpp"):
        self.whisper_path = Path(whisper_path)
        self.streaming_model = "ggml-tiny.en.bin"  # Fast, small model
        self.refining_model = "ggml-base.en.bin"   # Accurate, larger model
        
        # Verify whisper.cpp installation
        if not self.whisper_path.exists():
            raise FileNotFoundError(f"whisper.cpp not found at {whisper_path}")
        
        # Verify models exist
        self._verify_models()
        
        # Transcription state
        self.is_streaming = False
        self.streaming_queue = queue.Queue()
        self.current_transcript = []
        
    def _verify_models(self):
        """Verify that required models are available"""
        tiny_model = self.whisper_path / "models" / self.streaming_model
        base_model = self.whisper_path / "models" / self.refining_model
        
        if not tiny_model.exists():
            print(f"Warning: Streaming model not found at {tiny_model}")
            print("Downloading tiny model...")
            self._download_model("tiny.en")
            
        if not base_model.exists():
            print(f"Warning: Refining model not found at {base_model}")
            print("Downloading base model...")
            self._download_model("base.en")
    
    def _download_model(self, model_name: str):
        """Download a Whisper model"""
        download_script = self.whisper_path / "models" / "download-ggml-model.sh"
        if download_script.exists():
            try:
                subprocess.run([str(download_script), model_name], 
                             cwd=self.whisper_path / "models", 
                             check=True, capture_output=True)
                print(f"Successfully downloaded {model_name} model")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {model_name} model: {e}")
                sys.exit(1)
        else:
            print(f"Download script not found at {download_script}")
            sys.exit(1)
    
    def _run_whisper(self, model: str, input_file: str, output_file: str = None, 
                     options: List[str] = None) -> str:
        """Run whisper.cpp with specified parameters"""
        cmd = [
            str(self.whisper_path / "build" / "bin" / "whisper-cli"),
            "-m", str(self.whisper_path / "models" / model),
            "-f", input_file,
            "-otxt"
        ]
        

        
        if output_file:
            cmd.extend(["-of", output_file])
        
        if options:
            cmd.extend(options)
        
        # Set library path for dynamic library loading
        env = os.environ.copy()
        lib_paths = [
            str(self.whisper_path / "build" / "src"),
            str(self.whisper_path / "build" / "ggml" / "src"),
            str(self.whisper_path / "build" / "ggml" / "src" / "ggml-metal"),
            str(self.whisper_path / "build" / "ggml" / "src" / "ggml-blas")
        ]
        env["DYLD_LIBRARY_PATH"] = ":".join(lib_paths)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            
            # If output file was specified, read from it
            if output_file and os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Otherwise return stdout
                return result.stdout
                
        except subprocess.CalledProcessError as e:
            print(f"Whisper transcription failed: {e}")
            print(f"stderr: {e.stderr}")
            return ""
        except Exception as e:
            print(f"Unexpected error: {e}")
            return ""
    
    def streaming_pass(self, audio_file: str, output_file: str = None) -> str:
        """
        First pass: Fast streaming transcription using tiny model
        Returns the transcript text
        """
        print("Starting streaming pass with tiny model...")
        
        # Use tiny model for speed
        options = [
            "--no-timestamps",  # Faster without timestamp processing
            "--beam-size", "1",    # Use greedy decoding
            "--threads", "4"       # Optimize for speed
        ]
        
        transcript = self._run_whisper(self.streaming_model, audio_file, output_file, options)
        
        if transcript:
            print("Streaming pass completed successfully")
            return transcript
        else:
            print("Streaming pass failed")
            return ""
    
    def refining_pass(self, audio_file: str, output_file: str = None, 
                     initial_transcript: str = None) -> str:
        """
        Second pass: High-accuracy transcription using base model
        Can use initial transcript as context for better accuracy
        """
        print("Starting refining pass with base model...")
        
        # Use base model for accuracy
        options = [
            "--beam-size", "5",    # Use beam search for better results
            "--threads", "8",      # More threads for accuracy
            "--word-thold", "0.01", # Lower threshold for more words
            "--entropy-thold", "2.4" # Lower entropy threshold
        ]
        
        # If we have an initial transcript, create a prompt file
        if initial_transcript:
            prompt_file = self._create_prompt_file(initial_transcript)
            if prompt_file:
                options.extend(["--prompt", prompt_file])
        
        transcript = self._run_whisper(self.refining_model, audio_file, output_file, options)
        
        # Clean up prompt file
        if initial_transcript and 'prompt_file' in locals():
            os.unlink(prompt_file)
        
        if transcript:
            print("Refining pass completed successfully")
            return transcript
        else:
            print("Refining pass failed")
            return ""
    
    def _create_prompt_file(self, transcript: str) -> Optional[str]:
        """Create a temporary prompt file for whisper.cpp"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(transcript)
                return f.name
        except Exception as e:
            print(f"Failed to create prompt file: {e}")
            return None
    
    def two_pass_transcribe(self, audio_file: str, output_file: str = None) -> str:
        """
        Perform two-pass transcription: streaming then refining
        """
        print(f"Starting two-pass transcription of {audio_file}")
        
        # First pass: Fast transcription
        streaming_result = self.streaming_pass(audio_file)
        if not streaming_result:
            print("Streaming pass failed, attempting direct refining pass...")
            return self.refining_pass(audio_file, output_file)
        
        # Second pass: Accurate transcription with streaming result as context
        final_result = self.refining_pass(audio_file, output_file, streaming_result)
        
        if final_result:
            # Save both results
            if output_file:
                base_name = Path(output_file).stem
                streaming_file = f"{base_name}_streaming.txt"
                with open(streaming_file, 'w') as f:
                    f.write(streaming_result)
                print(f"Streaming result saved to {streaming_file}")
            
            return final_result
        else:
            print("Both passes failed, returning streaming result")
            return streaming_result
    
    def resume_transcription(self, audio_file: str, partial_file: str, 
                           output_file: str = None, start_time: float = 0.0) -> str:
        """
        Resume transcription from a specific point using partial transcript
        """
        print(f"Resuming transcription from {start_time}s with partial transcript")
        
        # Read partial transcript
        try:
            with open(partial_file, 'r') as f:
                partial_transcript = f.read()
        except FileNotFoundError:
            print(f"Partial transcript file {partial_file} not found")
            return ""
        
        # Create a prompt with the partial transcript
        prompt = f"Continuing from timestamp {start_time}s: {partial_transcript}"
        
        # Use refining model with the partial transcript as context
        options = [
            "--prompt", self._create_prompt_file(prompt),
            "--offset-t", str(int(start_time * 1000)),  # Offset in milliseconds
            "--max-tokens", "0",
            "--beam-size", "5"
        ]
        
        transcript = self._run_whisper(self.refining_model, audio_file, output_file, options)
        
        if transcript:
            print("Resumed transcription completed successfully")
            return transcript
        else:
            print("Resumed transcription failed")
            return ""
    
    def save_transcript(self, transcript: str, output_file: str, format: str = "txt"):
        """Save transcript in specified format"""
        if format == "txt":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript)
        elif format == "md":
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Lecture Transcript\n\n{transcript}")
        elif format == "json":
            data = {
                "transcript": transcript,
                "timestamp": time.time(),
                "model": "two-pass-whisper"
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        print(f"Transcript saved to {output_file}")
    
    def get_audio_duration(self, audio_file: str) -> float:
        """Get audio file duration using ffprobe"""
        try:
            cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                   "-of", "csv=p=0", audio_file]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            print("Warning: Could not determine audio duration")
            return 0.0

def main():
    parser = argparse.ArgumentParser(description="Two-Pass Lecture Transcription System")
    parser.add_argument("--input", "-i", required=True, help="Input audio file")
    parser.add_argument("--output", "-o", help="Output transcript file")
    parser.add_argument("--stream", action="store_true", help="Streaming mode only")
    parser.add_argument("--refine", action="store_true", help="Refining mode only")
    parser.add_argument("--resume", action="store_true", help="Resume from partial transcript")
    parser.add_argument("--partial", help="Partial transcript file for resume mode")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds for resume mode")
    parser.add_argument("--format", choices=["txt", "md", "json"], default="txt", help="Output format")
    parser.add_argument("--whisper-path", default="./whisper.cpp", help="Path to whisper.cpp installation")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)
    
    # Initialize transcriber
    try:
        transcriber = TwoPassTranscriber(args.whisper_path)
    except Exception as e:
        print(f"Error initializing transcriber: {e}")
        sys.exit(1)
    
    # Determine output file
    if not args.output:
        base_name = Path(args.input).stem
        args.output = f"{base_name}_transcript.{args.format}"
    
    # Perform transcription based on mode
    if args.stream:
        print("Streaming mode: Fast transcription only")
        transcript = transcriber.streaming_pass(args.input)
    elif args.refine:
        print("Refining mode: Accurate transcription only")
        transcript = transcriber.refining_pass(args.input)
    elif args.resume:
        if not args.partial:
            print("Error: --partial file required for resume mode")
            sys.exit(1)
        print("Resume mode: Continue from partial transcript")
        transcript = transcriber.resume_transcription(
            args.input, args.partial, args.output, args.start_time
        )
    else:
        print("Two-pass mode: Streaming then refining")
        transcript = transcriber.two_pass_transcribe(args.input, args.output)
    
    if transcript:
        # Save transcript
        transcriber.save_transcript(transcript, args.output, args.format)
        
        # Print summary
        duration = transcriber.get_audio_duration(args.input)
        if duration > 0:
            print(f"Audio duration: {duration:.2f} seconds")
        
        print(f"Transcription completed successfully!")
        print(f"Output saved to: {args.output}")
    else:
        print("Transcription failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
