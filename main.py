#!/usr/bin/env python3
"""
Main application for the Lecture Transcriber.

This script provides a real-time transcription UI using a two-pass system.
- Pass 1: A small, fast model for live transcription.
- Pass 2: A larger, more accurate model to refine the transcript in the background.

The UI is built using curses to provide a simple, scrollable view.
"""

import argparse
import curses
import os
import sys
import time
import threading
import queue
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List

try:
    import pyaudio
    import numpy as np
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    
# Local imports (make sure these files exist)
# These are not needed anymore as the logic is integrated here.
# from two_pass_transcriber import TwoPassTranscriber
# from live_streaming_transcriber import LiveStreamingTranscriber


class Transcriber:
    """
    Handles audio processing and transcription logic.
    """
    def __init__(self, args, ui):
        self.args = args
        self.ui = ui
        self.whisper_path = Path(args.whisper_path)
        self.streaming_model = "ggml-tiny.en.bin"
        self.refining_model = "ggml-base.en.bin"
        self.is_running = False
        
        self.audio_queue = queue.Queue()
        self.refine_queue = queue.Queue()
        
        self.sample_rate = 16000
        self.chunk_duration = 5.0  # seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)

        self.audio_buffer = []
        if PYAUDIO_AVAILABLE:
            self.pyaudio_instance = pyaudio.PyAudio()
            self.pyaudio_stream = None

        self.session_name = args.session if args.session else self._get_default_session_name()
        self.transcripts_dir = Path("./transcripts")
        self.sessions_dir = Path("./sessions")
        self.transcripts_dir.mkdir(exist_ok=True)
        self.sessions_dir.mkdir(exist_ok=True)

    def _get_default_session_name(self):
        """Generate a default session name based on input or current time."""
        if self.args.input:
            return Path(self.args.input).stem
        else:
            return f"mic_session_{time.strftime('%Y%m%d-%H%M%S')}"

    def save_session(self):
        """Save the current transcription history to a session file."""
        session_path = self.sessions_dir / f"{self.session_name}.session"
        with open(session_path, 'w') as f:
            for line in self.ui.history:
                f.write(f"{line}\n")
    
    def load_session(self):
        """Load a transcription history from a session file."""
        session_path = self.sessions_dir / f"{self.session_name}.session"
        if session_path.exists():
            with open(session_path, 'r') as f:
                for line in f:
                    self.ui.history.append(line.strip())
            # Redraw history from loaded data
            for i, line in enumerate(self.ui.history):
                self.ui.history_pad.addstr(i, 0, line)
            self.ui.draw_history()

    def start(self):
        """Start the transcription process."""
        if self.args.resume:
            self.load_session()
        self.is_running = True
        
        # Start threads for each part of the process
        self.audio_thread = threading.Thread(target=self.audio_processor)
        self.streaming_thread = threading.Thread(target=self.streaming_processor)
        self.refining_thread = threading.Thread(target=self.refining_processor)

        self.audio_thread.start()
        self.streaming_thread.start()
        self.refining_thread.start()

    def stop(self):
        """Stop the transcription process and save the final transcript."""
        self.is_running = False
        # Wait for threads to finish
        if self.audio_thread: self.audio_thread.join()
        if self.streaming_thread: self.streaming_thread.join()
        if self.refining_thread: self.refining_thread.join()
        self.save_final_transcript()

    def _run_whisper(self, model: str, input_file: str, options: List[str] = None) -> str:
        """Helper to run whisper.cpp."""
        cmd = [
            str(self.whisper_path / "build" / "bin" / "whisper-cli"),
            "-m", str(self.whisper_path / "models" / model),
            "-f", input_file,
            "-otxt",
        ]
        if options:
            cmd.extend(options)

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
            # whisper-cli with -otxt doesn't print to stdout, it creates a file.
            output_path = Path(f"{input_file}.txt")
            if output_path.exists():
                transcript = output_path.read_text().strip()
                output_path.unlink()
                return transcript
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.ui.add_to_history(f"Whisper failed: {e.stderr}", "ERROR")
            return ""

    def audio_processor(self):
        """Handles audio input from mic or file."""
        if self.args.mic:
            self._mic_input()
        elif self.args.input:
            self._file_input()
        self.audio_queue.put(None)

    def _mic_input(self):
        # Mic input logic from live_streaming_transcriber.py
        if not PYAUDIO_AVAILABLE:
            self.ui.add_to_history("Pyaudio not available.", "ERROR")
            return

        self.pyaudio_stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16, channels=1, rate=self.sample_rate,
            input=True, frames_per_buffer=1024,
            stream_callback=self._mic_callback
        )
        self.pyaudio_stream.start_stream()
        while self.is_running and self.pyaudio_stream.is_active():
            time.sleep(0.1)
        self.pyaudio_stream.stop_stream()
        self.pyaudio_stream.close()

    def _mic_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.extend(audio_data)
        while len(self.audio_buffer) >= self.chunk_samples:
            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.chunk_samples:]
            # Convert to float32
            float_chunk = np.array(chunk).astype(np.float32) / 32768.0
            self.audio_queue.put(float_chunk)
        return (in_data, pyaudio.paContinue)

    def _file_input(self):
        # File input logic from live_streaming_transcriber.py
        cmd = [
            "ffmpeg", "-i", self.args.input,
            "-f", "s16le", "-ac", "1", "-ar", str(self.sample_rate), "-"
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunk_size = self.chunk_samples * 2 # 16-bit audio
        while self.is_running:
            chunk_data = process.stdout.read(chunk_size)
            if not chunk_data:
                break
            
            audio_chunk = np.frombuffer(chunk_data, dtype=np.int16)
            float_chunk = audio_chunk.astype(np.float32) / 32768.0
            self.audio_queue.put(float_chunk)
        process.wait()

    def _transcribe_chunk(self, model, audio_chunk, options=None, prompt=None):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        
        if not SOUNDFILE_AVAILABLE:
            self.ui.add_to_history("Soundfile not installed. Cannot write WAV.", "ERROR")
            return ""
        
        sf.write(wav_path, audio_chunk, self.sample_rate)
        
        if prompt:
            options = (options or []) + ["--prompt", prompt]

        transcript = self._run_whisper(model, wav_path, options)
        Path(wav_path).unlink()
        return transcript

    def streaming_processor(self):
        """Processes audio chunks with the streaming model."""
        history_index = 0
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1)
                if chunk is None:
                    self.refine_queue.put(None)
                    break
                
                options = ["--no-timestamps", "--beam-size", "1"]
                transcript = self._transcribe_chunk(self.streaming_model, chunk, options)
                
                if transcript:
                    self.ui.update_live_text(transcript)
                    self.ui.add_to_history(transcript, "Streaming")
                    self.refine_queue.put((history_index, chunk, transcript))
                    history_index += 1

            except queue.Empty:
                continue

    def refining_processor(self):
        """Processes streaming results with the refining model."""
        while self.is_running:
            try:
                job = self.refine_queue.get(timeout=1)
                if job is None:
                    break
                
                index, chunk, stream_transcript = job
                
                options = ["--beam-size", "5", "--word-thold", "0.01"]
                refined_transcript = self._transcribe_chunk(
                    self.refining_model, chunk, options, prompt=stream_transcript
                )
                
                if refined_transcript:
                    self.ui.update_history_line(index + 1, refined_transcript, "Refined")

            except queue.Empty:
                continue

    def save_final_transcript(self):
        """Save the complete, refined transcript."""
        transcript_path = self.transcripts_dir / f"{self.session_name}.txt"
        with open(transcript_path, 'w') as f:
            f.write(f"# Final Transcript for session: {self.session_name}\n\n")
            for line in self.ui.history:
                # We could filter here for only 'Refined' lines if needed
                f.write(f"{line}\n")
        self.ui.add_to_history(f"Final transcript saved to {transcript_path}", "System")


class TranscriptionUI:
    """
    Manages the curses-based UI for the transcriber.
    """
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.history = []
        self.scroll_pos = 0
        self.setup_windows()

    def setup_windows(self):
        """Set up the curses windows for the UI."""
        self.stdscr.nodelay(True)
        curses.curs_set(0)
        self.height, self.width = self.stdscr.getmaxyx()

        # Window for current transcription
        self.live_win = curses.newwin(5, self.width, self.height - 5, 0)
        self.live_win.box()
        self.live_win.addstr(1, 2, "Live Transcription:")

        # Window for scrollable history
        self.history_pad = curses.newpad(1000, self.width - 2)
        self.history_win_height = self.height - 6
        
        self.stdscr.addstr(0, 2, "Transcription History (Scroll with Up/Down arrows)")
        self.stdscr.refresh()

    def update_live_text(self, text: str):
        """Update the live transcription window with new text."""
        self.live_win.clear()
        self.live_win.box()
        self.live_win.addstr(1, 2, "Live Transcription:")
        self.live_win.addstr(2, 4, text[:self.width-6])
        self.live_win.refresh()

    def add_to_history(self, text: str, status: str = "Streaming"):
        """Add a new line to the history."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] [{status}] {text}"
        self.history.append(line)
        self.history_pad.addstr(len(self.history) - 1, 0, line)
        self.draw_history()

    def update_history_line(self, index: int, new_text: str, new_status: str):
        """Update a specific line in the history."""
        if 0 <= index < len(self.history):
            timestamp = self.history[index].split(']')[0][1:]
            line = f"[{timestamp}] [{new_status}] {new_text}"
            self.history[index] = line
            # Clear line before redrawing
            self.history_pad.addstr(index, 0, " " * (self.width - 3))
            self.history_pad.addstr(index, 0, line)
            self.draw_history()

    def draw_history(self):
        """Draw the scrollable history pad."""
        self.stdscr.refresh()
        self.history_pad.refresh(self.scroll_pos, 0, 1, 1, self.history_win_height, self.width - 2)

    def handle_input(self):
        """Handle user input for scrolling."""
        try:
            key = self.stdscr.getch()
            if key == curses.KEY_UP:
                self.scroll_pos = max(0, self.scroll_pos - 1)
                self.draw_history()
            elif key == curses.KEY_DOWN:
                if len(self.history) > self.history_win_height:
                    self.scroll_pos = min(len(self.history) - self.history_win_height, self.scroll_pos + 1)
                    self.draw_history()
            # Add a quit key
            elif key == ord('q'):
                return False
        except curses.error:
            pass # No input
        return True

def main_ui(stdscr, args):
    """The main function to run the transcriber with the Curses UI."""
    try:
        ui = TranscriptionUI(stdscr)
        transcriber = Transcriber(args, ui)
        
        ui.add_to_history("System Initialized.")
        ui.update_live_text("Starting transcription...")
        
        transcriber.start()

        # Main loop for UI
        try:
            while transcriber.is_running:
                if not ui.handle_input():
                    transcriber.is_running = False # Signal threads to stop
                    break
                time.sleep(0.1)
                # Periodically save session
                transcriber.save_session()
        except KeyboardInterrupt:
            ui.add_to_history("Interrupted by user", "System")
        finally:
            transcriber.stop()
            ui.add_to_history("Transcription finished.", "System")
            # Keep UI open for a moment to show final status
            ui.update_live_text("Press 'q' to quit...")
            while ui.handle_input():
                time.sleep(0.1)
    except Exception as e:
        curses.endwin()
        print(f"UI Error: {e}")
        raise

if __name__ == '__main__':
    print(f"Python executable: {sys.executable}")
    parser = argparse.ArgumentParser(description="Real-time Lecture Transcriber")
    parser.add_argument("--input", "-i", help="Input audio file for transcription")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    parser.add_argument("--whisper-path", default="./whisper.cpp", 
                       help="Path to whisper.cpp installation")
    parser.add_argument("--session", help="Specify a session name for saving and resuming.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous session.")
    
    args = parser.parse_args()

    if not args.input and not args.mic:
        print("Error: Must specify either --input or --mic")
        sys.exit(1)

    if args.mic and not PYAUDIO_AVAILABLE:
        print("Error: --mic option requires PyAudio, but it's not installed.")
        print("Please install it via 'pip install -r requirements.txt'")
        sys.exit(1)

    curses.wrapper(main_ui, args)
