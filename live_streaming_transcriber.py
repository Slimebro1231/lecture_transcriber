#!/usr/bin/env python3
"""
Live Streaming Lecture Transcriber

This provides real-time transcription during lectures with:
- Live scrolling transcript display
- Audio input from microphone or file
- Configurable update intervals
- Save functionality for later refinement

Usage:
    python live_streaming_transcriber.py --mic                    # Use microphone
    python live_streaming_transcriber.py --input audio_file.wav   # Use audio file
    python live_streaming_transcriber.py --save live_transcript.txt  # Save output
"""

import argparse
import os
import sys
import time
import threading
import queue
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List
import signal
import atexit

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: pyaudio not available. Microphone input disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Audio processing disabled.")

class LiveStreamingTranscriber:
    def __init__(self, whisper_path: str = "./whisper.cpp", 
                 model: str = "ggml-tiny.en.bin"):
        self.whisper_path = Path(whisper_path)
        self.model = model
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.current_transcript = []
        self.audio_buffer = []
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # seconds per chunk
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Verify whisper.cpp installation
        if not self.whisper_path.exists():
            raise FileNotFoundError(f"whisper.cpp not found at {whisper_path}")
        
        # Verify model exists
        self._verify_model()
        
        # Audio setup
        if PYAUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
            self.stream = None
        
        # Transcription state
        self.last_transcript = ""
        self.transcript_history = []
        self.max_history_lines = 20
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        atexit.register(self.cleanup)
    
    def _verify_model(self):
        """Verify that the streaming model is available"""
        model_path = self.whisper_path / "models" / self.model
        
        if not model_path.exists():
            print(f"Warning: Streaming model not found at {model_path}")
            print("Downloading tiny model...")
            self._download_model("tiny.en")
    
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
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print("\nShutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
    
    def start_microphone(self):
        """Start microphone input stream"""
        if not PYAUDIO_AVAILABLE:
            print("Error: pyaudio not available for microphone input")
            return False
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback
            )
            self.stream.start_stream()
            print("Microphone input started")
            return True
        except Exception as e:
            print(f"Failed to start microphone: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for microphone input"""
        if not NUMPY_AVAILABLE:
            return (in_data, pyaudio.paContinue)
            
        if self.is_running:
            # Convert bytes to float32
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.extend(audio_data)
            
            # Process chunks when we have enough samples
            while len(self.audio_buffer) >= self.chunk_samples:
                chunk = self.audio_buffer[:self.chunk_samples]
                self.audio_buffer = self.audio_buffer[self.chunk_samples:]
                self.audio_queue.put(chunk)
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio_file(self, audio_file: str):
        """Process audio file in chunks for live transcription"""
        if not NUMPY_AVAILABLE:
            print("Error: numpy not available for audio processing")
            return False
            
        if not os.path.exists(audio_file):
            print(f"Error: Audio file {audio_file} not found")
            return False
        
        # Use ffmpeg to convert to raw audio and read in chunks
        try:
            cmd = [
                "ffmpeg", "-i", audio_file,
                "-f", "s16le", "-ac", "1", "-ar", str(self.sample_rate),
                "-"
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            chunk_size = self.chunk_samples * 2  # 16-bit samples = 2 bytes
            
            while True:
                chunk_data = process.stdout.read(chunk_size)
                if not chunk_data:
                    break
                
                # Convert to float32
                audio_chunk = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if len(audio_chunk) == self.chunk_samples:
                    self.audio_queue.put(audio_chunk)
                
                time.sleep(0.1)  # Small delay to simulate real-time
            
            process.wait()
            return True
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            return False
    
    def transcribe_chunk(self, audio_chunk) -> str:
        """Transcribe a single audio chunk"""
        try:
            # Save chunk to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            # Convert to WAV using ffmpeg
            cmd = [
                "ffmpeg", "-f", "f32le", "-ac", "1", "-ar", str(self.sample_rate),
                "-i", "-", "-y", temp_wav_path
            ]
            
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Handle different audio chunk types
            if hasattr(audio_chunk, 'tobytes'):
                chunk_bytes = audio_chunk.tobytes()
            elif isinstance(audio_chunk, (list, tuple)):
                # Convert list/tuple to bytes
                import struct
                chunk_bytes = struct.pack('f' * len(audio_chunk), *audio_chunk)
            else:
                chunk_bytes = audio_chunk
                
            process.communicate(input=chunk_bytes)
            
            if process.returncode != 0:
                return ""
            
            # Transcribe with whisper.cpp
            whisper_cmd = [
                str(self.whisper_path / "build" / "bin" / "whisper-cli"),
                "-m", str(self.whisper_path / "models" / self.model),
                "-f", temp_wav_path,
                "-otxt",
                "--no-timestamps",
                "--beam-size", "1",
                "--threads", "4"
            ]
            
            # Set library path for dynamic library loading
            env = os.environ.copy()
            lib_paths = [
                str(self.whisper_path / "build" / "src"),
                str(self.whisper_path / "build" / "ggml" / "src"),
                str(self.whisper_path / "build" / "ggml" / "src" / "ggml-metal"),
                str(self.whisper_path / "build" / "ggml" / "src" / "ggml-blas")
            ]
            env["DYLD_LIBRARY_PATH"] = ":".join(lib_paths)
            
            result = subprocess.run(whisper_cmd, capture_output=True, text=True, check=True, env=env)
            
            # Read from output file if it exists, otherwise use stdout
            transcript = ""
            if os.path.exists(temp_wav_path + ".txt"):
                with open(temp_wav_path + ".txt", 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
            else:
                transcript = result.stdout.strip()
            
            # Clean up temp file
            os.unlink(temp_wav_path)
            
            return transcript
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def audio_processor(self):
        """Process audio chunks and generate transcriptions"""
        while self.is_running:
            try:
                # Get audio chunk from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Transcribe chunk
                transcript = self.transcribe_chunk(audio_chunk)
                
                if transcript:
                    # Add to transcript history
                    timestamp = time.strftime("%H:%M:%S")
                    transcript_line = f"[{timestamp}] {transcript}"
                    self.transcript_history.append(transcript_line)
                    
                    # Keep only recent history
                    if len(self.transcript_history) > self.max_history_lines:
                        self.transcript_history.pop(0)
                    
                    # Update display
                    self._update_display()
                    
                    # Put in transcript queue for saving
                    self.transcript_queue.put(transcript)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processing error: {e}")
    
    def _update_display(self):
        """Update the live transcript display"""
        # Clear screen (works on most terminals)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print("LIVE LECTURE TRANSCRIPTION")
        print("=" * 80)
        print(f"Model: {self.model}")
        print(f"Status: {'Running' if self.is_running else 'Stopped'}")
        print(f"Audio chunks processed: {len(self.transcript_history)}")
        print("=" * 80)
        print()
        
        # Display transcript history
        for line in self.transcript_history:
            print(line)
        
        print()
        print("=" * 80)
        print("Press Ctrl+C to stop")
        print("=" * 80)
    
    def save_transcript(self, output_file: str):
        """Save the complete transcript"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("LIVE LECTURE TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")
                
                for line in self.transcript_history:
                    f.write(line + "\n")
            
            print(f"Transcript saved to {output_file}")
            return True
        except Exception as e:
            print(f"Failed to save transcript: {e}")
            return False
    
    def start(self, input_source: str = None):
        """Start the live transcription system"""
        self.is_running = True
        
        # Start audio input
        if input_source:
            if os.path.exists(input_source):
                print(f"Starting file-based transcription: {input_source}")
                file_thread = threading.Thread(target=self.process_audio_file, args=(input_source,))
                file_thread.daemon = True
                file_thread.start()
            else:
                print(f"Error: Input source {input_source} not found")
                return False
        else:
            if PYAUDIO_AVAILABLE:
                if not self.start_microphone():
                    return False
            else:
                print("Error: No audio input available")
                return False
        
        # Start audio processing thread
        audio_thread = threading.Thread(target=self.audio_processor)
        audio_thread.daemon = True
        audio_thread.start()
        
        print("Live transcription started")
        self._update_display()
        
        # Main loop
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the live transcription system"""
        self.is_running = False
        print("\nStopping live transcription...")
        
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        print("Live transcription stopped")

def main():
    parser = argparse.ArgumentParser(description="Live Streaming Lecture Transcriber")
    parser.add_argument("--input", "-i", help="Input audio file for transcription")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--output", "-o", help="Output transcript file")
    parser.add_argument("--model", "-m", default="models/ggml-tiny.en.bin", 
                       help="Whisper model to use")
    parser.add_argument("--whisper-path", default="./whisper.cpp", 
                       help="Path to whisper.cpp installation")
    parser.add_argument("--chunk-duration", type=float, default=3.0,
                       help="Audio chunk duration in seconds")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input and not args.mic:
        print("Error: Must specify either --input or --mic")
        sys.exit(1)
    
    if args.input and args.mic:
        print("Error: Cannot use both --input and --mic")
        sys.exit(1)
    
    # Initialize transcriber
    try:
        transcriber = LiveStreamingTranscriber(args.whisper_path, args.model)
        transcriber.chunk_duration = args.chunk_duration
    except Exception as e:
        print(f"Error initializing transcriber: {e}")
        sys.exit(1)
    
    # Start transcription
    try:
        transcriber.start(args.input)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        transcriber.stop()
        
        # Save transcript if requested
        if args.output:
            transcriber.save_transcript(args.output)

if __name__ == "__main__":
    main()
