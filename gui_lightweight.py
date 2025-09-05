#!/usr/bin/env python3
"""
Lightweight GUI for the transcription system with memory optimizations.
"""

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import time
from engine_lightweight import LightweightTranscriptionEngine

class LightweightTranscriberGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Lecture Transcriber (Lightweight)")
        self.root.geometry("800x600")
        
        # Configure dark theme
        self.root.configure(bg='#2b2b2b')
        
        # Create main frame
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            main_frame, 
            text="Initializing...", 
            bg='#2b2b2b', 
            fg='white',
            font=('Arial', 12)
        )
        self.status_label.pack(pady=(0, 10))
        
        # Current transcription label
        current_frame = tk.Frame(main_frame, bg='#2b2b2b')
        current_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            current_frame, 
            text="Current Speech:", 
            bg='#2b2b2b', 
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(anchor=tk.W)
        
        self.current_text_label = tk.Label(
            current_frame,
            text="Listening...",
            bg='#2b2b2b',
            fg='#87CEEB',  # Light blue
            font=('Courier', 11),
            wraplength=750,
            justify=tk.LEFT
        )
        self.current_text_label.pack(anchor=tk.W, pady=(5, 0))
        
        # History section
        history_frame = tk.Frame(main_frame, bg='#2b2b2b')
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            history_frame, 
            text="Transcription History:", 
            bg='#2b2b2b', 
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(anchor=tk.W)
        
        # Create scrolled text widget
        self.history_text = scrolledtext.ScrolledText(
            history_frame,
            bg='#1e1e1e',
            fg='white',
            font=('Courier', 10),
            wrap=tk.WORD,
            height=15
        )
        self.history_text.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Configure text tags for different statuses
        self.history_text.tag_configure("transcribed", foreground="#90EE90")  # Light green
        
        # Control buttons
        button_frame = tk.Frame(main_frame, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = tk.Button(
            button_frame,
            text="Start Transcription",
            command=self.start_transcription,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10),
            padx=20
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = tk.Button(
            button_frame,
            text="Stop Transcription",
            command=self.stop_transcription,
            bg='#f44336',
            fg='white',
            font=('Arial', 10),
            padx=20,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = tk.Button(
            button_frame,
            text="Clear History",
            command=self.clear_history,
            bg='#FF9800',
            fg='white',
            font=('Arial', 10),
            padx=20
        )
        self.clear_button.pack(side=tk.LEFT)
        
        # Memory usage label
        self.memory_label = tk.Label(
            button_frame,
            text="Memory: --",
            bg='#2b2b2b',
            fg='yellow',
            font=('Arial', 9)
        )
        self.memory_label.pack(side=tk.RIGHT)
        
        # Initialize engine
        self.engine = LightweightTranscriptionEngine(self.update_callback)
        self.is_running = False
        
        # Start memory monitoring
        self.start_memory_monitoring()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def start_memory_monitoring(self):
        """Start monitoring memory usage."""
        def monitor():
            while True:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.root.after(0, lambda: self.memory_label.config(text=f"Memory: {memory_mb:.1f} MB"))
                except:
                    pass
                time.sleep(5)  # Update every 5 seconds
        
        threading.Thread(target=monitor, daemon=True).start()
    
    def start_transcription(self):
        """Start the transcription engine."""
        if not self.is_running:
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.engine.start()
    
    def stop_transcription(self):
        """Stop the transcription engine."""
        if self.is_running:
            self.is_running = False
            self.engine.stop()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.update_callback("status", "Transcription stopped")
    
    def clear_history(self):
        """Clear the transcription history."""
        self.history_text.delete(1.0, tk.END)
        self.current_text_label.config(text="Listening...")
    
    def update_callback(self, message_type, data):
        """Handle updates from the transcription engine."""
        if message_type == "status":
            self.root.after(0, lambda: self.status_label.config(text=data))
        elif message_type == "transcript":
            self.root.after(0, lambda: self.update_transcript_display(data))
    
    def update_transcript_display(self, transcript_data):
        """Update the transcript display."""
        text = transcript_data["text"]
        status = transcript_data["status"]
        
        # Update current speech
        self.current_text_label.config(text=text)
        
        # Add to history
        timestamp = time.strftime("%H:%M:%S")
        self.history_text.insert(tk.END, f"[{timestamp}] {text}\n")
        
        # Apply color tag
        start_line = self.history_text.index(tk.END + "-2l")
        end_line = self.history_text.index(tk.END + "-1l")
        self.history_text.tag_add("transcribed", start_line, end_line)
        
        # Auto-scroll to bottom
        self.history_text.see(tk.END)
    
    def on_closing(self):
        """Handle window closing."""
        if self.is_running:
            self.stop_transcription()
        self.root.destroy()
    
    def run(self):
        """Start the GUI."""
        self.root.mainloop()

if __name__ == "__main__":
    app = LightweightTranscriberGUI()
    app.run()
