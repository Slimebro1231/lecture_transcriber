#!/usr/bin/env python3
"""
Advanced Transcription System with GUI

This application provides a modern GUI for a two-pass transcription system
using state-of-the-art local ASR models from Hugging Face.
"""

import tkinter as tk
from tkinter import scrolledtext, font
import queue
from engine import TranscriptionEngine

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lecture Transcriber")
        self.root.geometry("800x600")

        self.setup_styles()
        self.create_widgets()

        self.transcription_queue = queue.Queue()
        self.transcripts = {} # Use a dict to easily update lines
        self.engine = TranscriptionEngine(self.update_callback)

    def setup_styles(self):
        """Configure fonts and colors for the UI."""
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Helvetica", size=11)
        
        self.title_font = font.Font(family="Helvetica", size=14, weight="bold")
        self.current_font = font.Font(family="Helvetica", size=18, weight="bold")

        self.root.configure(bg="#2E2E2E")
        self.colors = {
            "bg": "#2E2E2E", "fg": "#FFFFFF",
            "history_bg": "#1C1C1C", "current_bg": "#3C3C3C",
            "streaming": "#4CAF50", "refined": "#2196F3"
        }

    def create_widgets(self):
        """Create the main UI elements."""
        main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        history_label = tk.Label(main_frame, text="Finalized Transcript (Scrollable)", 
                                 font=self.title_font, bg=self.colors["bg"], fg=self.colors["fg"])
        history_label.pack(fill=tk.X)

        self.history_text = scrolledtext.ScrolledText(
            main_frame, 
            wrap=tk.WORD, 
            state='disabled', 
            bg=self.colors["history_bg"], 
            fg=self.colors["fg"],
            font=("Courier", 14), # Monospace font
            borderwidth=0,
            highlightthickness=0
        )
        self.history_text.pack(padx=10, pady=5, expand=True, fill='both')

        current_label = tk.Label(main_frame, text="Live Speech (Most Recent)", 
                                 font=self.title_font, bg=self.colors["bg"], fg=self.colors["fg"])
        current_label.pack(fill=tk.X, pady=(10, 0))
        
        self.current_text_label = tk.Label(
            main_frame, 
            text="Initializing...", 
            bg=self.colors["current_bg"], 
            fg=self.colors["fg"], 
            font=("Courier", 14, "bold"), # Monospace font
            wraplength=780,
            justify=tk.LEFT
        )
        self.current_text_label.pack(padx=10, pady=5, fill='x')
        
        # Configure tags for refined text
        self.history_text.tag_configure("refined", foreground="lightgreen")
        self.history_text.tag_configure("streaming", foreground="lightblue")

    def update_callback(self, event_type, data):
        """Callback for the transcription engine to send data to the GUI."""
        self.transcription_queue.put((event_type, data))

    def process_queue(self):
        """Process updates from the engine in the main GUI thread."""
        try:
            while not self.transcription_queue.empty():
                event_type, data = self.transcription_queue.get_nowait()
                if event_type == "status":
                    self.current_text_label.config(text=data)
                elif event_type == "transcript":
                    self.update_transcript_display(data)
        finally:
            self.root.after(100, self.process_queue)

    def update_transcript_display(self, data):
        """Updates the transcript display in the GUI."""
        text = data.get("text", "")
        status = data.get("status", "Streaming")
        transcript_id = data.get("id", 0)


        if status == "Refined":
            # Add refined text to the scrollable log
            self.history_text.configure(state='normal')
            self.history_text.insert('end', f"{text}\n\n")
            self.history_text.configure(state='disabled')
            self.history_text.see('end')
            # Clear the current streaming text
            self.current_text_label.config(text="")
        else:
            # Always show the most recent streaming text
            self.current_text_label.config(text=text)
        
        self.transcripts[transcript_id] = data

    def start(self):
        """Start the application."""
        self.engine.start()
        self.root.after(100, self.process_queue)
        
    
    def on_closing(self):
        """Handle window closing event."""
        self.engine.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    app.start()
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
