#!/usr/bin/env python3
"""
Lightweight transcription engine with memory optimizations.
Uses smaller models and aggressive memory management.
"""

import threading
import queue
import time
import numpy as np
import os
import signal
import sys
import gc
import psutil
from datetime import datetime

try:
    import pyaudio
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LightweightTranscriptionEngine:
    def __init__(self, update_callback):
        self.update_callback = update_callback
        
        # Use smaller, faster models to reduce memory usage
        self.model_id = "openai/whisper-tiny.en"  # Much smaller than large-v3
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

        # Single pipeline instead of two-pass
        self.pipeline = None

        # Audio settings - shorter chunks to reduce memory
        self.sample_rate = 16000
        self.chunk_duration = 5  # Shorter chunks (5 seconds instead of 15)
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        # Single queue with size limit
        self.audio_queue = queue.Queue(maxsize=3)  # Very small queue
        
        # Session persistence
        self.transcripts_folder = "transcripts"
        if not os.path.exists(self.transcripts_folder):
            os.makedirs(self.transcripts_folder)
        
        self.session_file = os.path.join(
            self.transcripts_folder, 
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self.session_transcripts = []
        
        # State
        self.is_running = False
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Aggressive memory management
        self.max_buffer_size = self.chunk_samples  # Only keep 1 chunk worth
        self.last_memory_check = time.time()
        self.memory_check_interval = 10  # Check every 10 seconds
        self.cleanup_interval = 60  # Force cleanup every 60 seconds
        self.last_cleanup = time.time()

        if AUDIO_AVAILABLE:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.input_device_index = None
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_model(self):
        """Load single lightweight model."""
        try:
            self.update_callback("status", "Initializing lightweight model (Whisper-Tiny)...")
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            self.update_callback("status", "✅ Model initialized. Starting microphone...")
        except Exception as e:
            self.update_callback("status", f"❌ Error initializing model: {e}")

    def start(self):
        """Start the transcription engine."""
        if not TRANSFORMERS_AVAILABLE or not AUDIO_AVAILABLE:
            self.update_callback("status", "❌ Missing critical dependencies.")
            return
        
        self.is_running = True
        
        threading.Thread(target=self._initialize_model, daemon=True).start()
        threading.Thread(target=self._audio_input_thread, daemon=True).start()
        threading.Thread(target=self._transcription_thread, daemon=True).start()

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals to save session before exit."""
        print(f"\nReceived signal {signum}. Saving session...")
        self.stop()
        sys.exit(0)
    
    def stop(self):
        self.is_running = False
        self._cleanup_memory()
        self._save_session()
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup."""
        try:
            # Clear audio buffer
            self.audio_buffer = np.array([], dtype=np.float32)
            
            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("🧹 Memory cleanup completed")
        except Exception as e:
            print(f"Error during memory cleanup: {e}")
    
    def _check_memory_usage(self):
        """Check and log memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > 500:  # Lower threshold for lightweight version
                print(f"⚠️  Memory usage: {memory_mb:.1f} MB")
                if memory_mb > 1000:  # Force cleanup at 1GB
                    print("🧹 Forcing memory cleanup...")
                    self._cleanup_memory()
            
            return memory_mb
        except Exception as e:
            print(f"Error checking memory: {e}")
            return 0
    
    def _save_session(self):
        """Save session data to TXT file."""
        try:
            with open(self.session_file, 'w') as f:
                f.write(f"Transcription Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                for transcript in self.session_transcripts:
                    f.write(f"[{transcript['status']}] {transcript['text']}\n\n")
            print(f"Session saved to {self.session_file}")
        except Exception as e:
            print(f"Error saving session: {e}")

    def _audio_input_thread(self):
        """Capture audio with aggressive memory management."""
        stream = None
        try:
            self.update_callback("status", "🎤 Initializing microphone...")
            stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32, 
                channels=1, 
                rate=self.sample_rate,
                input=True, 
                input_device_index=self.input_device_index,
                frames_per_buffer=1024,
                stream_callback=None
            )
            self.update_callback("status", "✅ Microphone ready. Listening...")
            
            while self.is_running:
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    new_data = np.frombuffer(data, dtype=np.float32)
                    
                    # Aggressive buffer management
                    if len(self.audio_buffer) > self.max_buffer_size:
                        # Keep only the most recent data
                        excess = len(self.audio_buffer) - self.max_buffer_size
                        self.audio_buffer = self.audio_buffer[excess:]
                    
                    self.audio_buffer = np.append(self.audio_buffer, new_data)
                    
                    if len(self.audio_buffer) >= self.chunk_samples:
                        chunk = self.audio_buffer[:self.chunk_samples].copy()
                        self.audio_buffer = self.audio_buffer[self.chunk_samples:]
                        
                        # Try to put in queue, skip if full
                        try:
                            self.audio_queue.put_nowait(chunk)
                        except queue.Full:
                            print("⚠️  Audio queue full, skipping chunk")
                            del chunk
                    
                    # Frequent memory checks
                    current_time = time.time()
                    if current_time - self.last_memory_check > self.memory_check_interval:
                        self._check_memory_usage()
                        self.last_memory_check = current_time
                    
                    # Periodic cleanup
                    if current_time - self.last_cleanup > self.cleanup_interval:
                        self._cleanup_memory()
                        self.last_cleanup = current_time
                        
                except Exception as e:
                    print(f"Audio read error: {e}")
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            self.update_callback("status", f"❌ Microphone Error: {e}")
            return
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass

    def _transcription_thread(self):
        """Single-pass transcription with memory management."""
        chunk_id = 0
        while self.is_running:
            if not self.pipeline:
                time.sleep(1)
                continue
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                
                # Process audio chunk
                result = self.pipeline(audio_chunk)
                transcript = result["text"].strip()
                
                if transcript and len(transcript) > 2:
                    transcript_data = {
                        "id": chunk_id, 
                        "text": transcript, 
                        "status": "Transcribed",
                        "timestamp": time.time()
                    }
                    
                    # Save to session data
                    self.session_transcripts.append(transcript_data)
                    self.update_callback("transcript", transcript_data)
                    chunk_id += 1
                
                # Explicitly delete audio chunk
                del audio_chunk
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {e}")
                continue
