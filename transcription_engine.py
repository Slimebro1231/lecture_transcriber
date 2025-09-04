import threading
import queue
import time
from pathlib import Path
import numpy as np

# Audio processing imports
try:
    import pyaudio
    import soundfile as sf
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Transformer imports
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class TranscriptionEngine:
    def __init__(self, update_callback):
        self.update_callback = update_callback
        
        # Models
        self.streaming_model_id = "distil-whisper/distil-large-v2"
        self.refining_model_id = "nvidia/canary-1b"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        # Queues
        self.audio_queue = queue.Queue()
        self.refine_queue = queue.Queue()
        
        # State
        self.is_running = False
        self.audio_buffer = np.array([], dtype=np.float32)

        if AUDIO_AVAILABLE:
            self.pyaudio_instance = pyaudio.PyAudio()

    def _initialize_models(self):
        """Load models on a background thread."""
        try:
            self.update_callback("status", "Initializing streaming model (Distil-Whisper)...")
            self.streaming_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.streaming_model_id,
                device=self.device
            )
            
            self.update_callback("status", "Initializing refining model (NVIDIA Canary)...")
            self.refining_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.refining_model_id,
                device=self.device
            )
            self.update_callback("status", "Models initialized. Starting microphone...")
        except Exception as e:
            self.update_callback("status", f"Error initializing models: {e}")

    def start(self):
        """Start the transcription engine."""
        if not TRANSFORMERS_AVAILABLE or not AUDIO_AVAILABLE:
            self.update_callback("status", "Missing critical dependencies (transformers, torch, or audio libraries).")
            return
        
        self.is_running = True
        
        # Initialize models in a separate thread to not freeze the GUI
        threading.Thread(target=self._initialize_models, daemon=True).start()
        
        # Start audio and processing threads
        self.audio_thread = threading.Thread(target=self._audio_input_thread, daemon=True)
        self.streaming_thread = threading.Thread(target=self._streaming_thread, daemon=True)
        self.refining_thread = threading.Thread(target=self._refining_thread, daemon=True)

        self.audio_thread.start()
        self.streaming_thread.start()
        self.refining_thread.start()

    def stop(self):
        """Stop the engine."""
        self.is_running = False

    def _audio_input_thread(self):
        """Capture audio from the microphone."""
        stream = self.pyaudio_instance.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024
        )
        while self.is_running:
            data = stream.read(1024)
            self.audio_buffer = np.append(self.audio_buffer, np.frombuffer(data, dtype=np.float32))
            
            if len(self.audio_buffer) >= self.chunk_samples:
                chunk = self.audio_buffer[:self.chunk_samples]
                self.audio_buffer = self.audio_buffer[self.chunk_samples:]
                self.audio_queue.put(chunk)
        stream.stop_stream()
        stream.close()

    def _streaming_thread(self):
        """Process audio with the streaming model."""
        chunk_id = 0
        while self.is_running:
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                if self.streaming_pipeline:
                    result = self.streaming_pipeline(audio_chunk.copy(), generate_kwargs={"task": "transcribe"})
                    transcript = result["text"]
                    
                    self.update_callback("transcript", {"id": chunk_id, "text": transcript, "status": "Streaming"})
                    self.refine_queue.put({"id": chunk_id, "audio": audio_chunk, "text": transcript})
                    chunk_id += 1
            except queue.Empty:
                continue

    def _refining_thread(self):
        """Process audio with the refining model."""
        while self.is_running:
            try:
                job = self.refine_queue.get(timeout=1)
                if self.refining_pipeline:
                    result = self.refining_pipeline(job["audio"].copy(), generate_kwargs={"task": "transcribe"})
                    transcript = result["text"]

                    self.update_callback("transcript", {"id": job["id"], "text": transcript, "status": "Refined"})
            except queue.Empty:
                continue
