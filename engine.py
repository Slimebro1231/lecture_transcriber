import threading
import queue
import time
import numpy as np
import os
import signal
import sys
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

class TranscriptionEngine:
    def __init__(self, update_callback):
        self.update_callback = update_callback
        
        # Mac-optimized models (English-only)
        self.streaming_model_id = "distil-whisper/distil-medium.en"
        self.refining_model_id = "openai/whisper-large-v3"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

        # Pipelines will be initialized later
        self.streaming_pipeline = None
        self.refining_pipeline = None

        # Audio settings - longer chunks for better sentence context
        self.sample_rate = 16000
        self.chunk_duration = 15  # seconds of audio per chunk (longer for sentences)
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        # Queues for multi-threading
        self.audio_queue = queue.Queue()
        self.refine_queue = queue.Queue()
        
        # Session persistence - simple TXT storage
        self.transcripts_folder = "transcripts"
        if not os.path.exists(self.transcripts_folder):
            os.makedirs(self.transcripts_folder)
        
        self.session_file = os.path.join(
            self.transcripts_folder, 
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self.session_transcripts = []
        
        # Sentence processing
        self.sentence_buffer = ""  # Accumulate text until we find sentence boundaries
        self.last_sentence_time = time.time()
        
        # State
        self.is_running = False
        self.audio_buffer = np.array([], dtype=np.float32)

        if AUDIO_AVAILABLE:
            self.pyaudio_instance = pyaudio.PyAudio()
            # Use system default microphone
            self.input_device_index = None
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _initialize_models(self):
        """Load models on a background thread to keep GUI responsive."""
        try:
            self.update_callback("status", "Initializing streaming model (Distil-Whisper)...")
            self.streaming_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.streaming_model_id,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            self.update_callback("status", "Initializing refining model (Whisper-Large-v3)...")
            self.refining_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.refining_model_id,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            self.update_callback("status", "✅ Models initialized. Starting microphone...")
        except Exception as e:
            self.update_callback("status", f"❌ Error initializing models: {e}")

    def start(self):
        """Start the transcription engine."""
        if not TRANSFORMERS_AVAILABLE or not AUDIO_AVAILABLE:
            self.update_callback("status", "❌ Missing critical dependencies.")
            return
        
        self.is_running = True
        
        threading.Thread(target=self._initialize_models, daemon=True).start()
        threading.Thread(target=self._audio_input_thread, daemon=True).start()
        threading.Thread(target=self._streaming_thread, daemon=True).start()
        threading.Thread(target=self._refining_thread, daemon=True).start()

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals to save session before exit."""
        print(f"\nReceived signal {signum}. Saving session...")
        self.stop()
        sys.exit(0)
    
    def stop(self):
        self.is_running = False
        self._save_session()
    
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
        """Capture audio from the microphone and put it into a queue."""
        stream = None
        try:
            self.update_callback("status", "🎤 Initializing microphone...")
            stream = self.pyaudio_instance.open(
                format=pyaudio.paFloat32, 
                channels=1, 
                rate=self.sample_rate,
                input=True, 
                input_device_index=self.input_device_index,  # Use system default
                frames_per_buffer=1024,
                stream_callback=None
            )
            self.update_callback("status", "✅ Microphone ready. Listening...")
            
            while self.is_running:
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    self.audio_buffer = np.append(self.audio_buffer, np.frombuffer(data, dtype=np.float32))
                    
                    if len(self.audio_buffer) >= self.chunk_samples:
                        chunk = self.audio_buffer[:self.chunk_samples]
                        self.audio_buffer = self.audio_buffer[self.chunk_samples:]
                        self.audio_queue.put(chunk)
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


    def _streaming_thread(self):
        """Process audio with sentence-aware streaming."""
        chunk_id = 0
        while self.is_running:
            if not self.streaming_pipeline:
                time.sleep(1)
                continue
            try:
                audio_chunk = self.audio_queue.get(timeout=1)
                result = self.streaming_pipeline(
                    audio_chunk.copy()
                )
                transcript = result["text"].strip()
                
                if transcript and len(transcript) > 2:
                    # Add to sentence buffer
                    self.sentence_buffer += " " + transcript
                    self.sentence_buffer = self.sentence_buffer.strip()
                    
                    # Check for sentence boundaries
                    sentences = self._extract_complete_sentences()
                    for sentence in sentences:
                        if sentence.strip():
                            self.update_callback("transcript", {"id": chunk_id, "text": sentence, "status": "Streaming"})
                            chunk_id += 1
                    
                    # Send to refining queue
                    self.refine_queue.put({"id": chunk_id, "audio": audio_chunk})
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Streaming error: {e}")
                continue
    
    def _extract_complete_sentences(self):
        """Extract complete sentences from the buffer."""
        sentences = []
        
        # Look for sentence endings
        sentence_endings = ['.', '!', '?']
        
        for ending in sentence_endings:
            while ending in self.sentence_buffer:
                end_pos = self.sentence_buffer.find(ending)
                if end_pos != -1:
                    sentence = self.sentence_buffer[:end_pos + 1].strip()
                    if len(sentence) > 10:  # Only keep substantial sentences
                        sentences.append(sentence)
                    self.sentence_buffer = self.sentence_buffer[end_pos + 1:].strip()
        
        return sentences

    def _refining_thread(self):
        """Process audio with the high-accuracy refining model."""
        while self.is_running:
            if not self.refining_pipeline:
                time.sleep(1)
                continue
            try:
                job = self.refine_queue.get(timeout=1)
                result = self.refining_pipeline(
                    job["audio"].copy()
                )
                transcript = result["text"].strip()
                
                if transcript and len(transcript) > 2: # Don't update with empty or very short transcripts
                    transcript_data = {
                        "id": job["id"], 
                        "text": transcript, 
                        "status": "Refined",
                        "timestamp": time.time()
                    }
                    # Save to session data
                    self.session_transcripts.append(transcript_data)
                    self.update_callback("transcript", transcript_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Refining error: {e}")
                continue
