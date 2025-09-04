#!/usr/bin/env python3
"""
Lecture Transcriber Launcher

A simple menu-driven interface for the lecture transcription system.
This provides a reliable UI that shows all available options.
"""

import os
import sys
import subprocess
from pathlib import Path

class TranscriptionLauncher:
    def __init__(self):
        self.whisper_path = Path("./whisper.cpp")
        self.sessions_dir = Path("./sessions")
        self.transcripts_dir = Path("./transcripts")
        
        # Ensure directories exist
        self.sessions_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def show_header(self):
        """Display the application header."""
        print("=" * 70)
        print("🎤 LECTURE TRANSCRIPTION SYSTEM")
        print("=" * 70)
        print("Two-Pass Transcription: Streaming + Refining")
        print("Real-time transcription with AI-optimized output")
        print("=" * 70)
        print()
    
    def show_menu(self):
        """Display the main menu options."""
        print("📋 MAIN MENU:")
        print()
        print("1. 🎙️  Start Live Microphone Transcription")
        print("2. 📁  Transcribe Audio File")
        print("3. 🔄  Resume Previous Session")
        print("4. 📊  View Session History")
        print("5. 📝  View Transcripts")
        print("6. 🧹  Clean Up Old Files")
        print("7. ⚙️   System Status & Test")
        print("8. ❌  Exit")
        print()
    
    def get_choice(self):
        """Get user menu choice."""
        try:
            choice = input("Select option (1-8): ").strip()
            return int(choice) if choice.isdigit() else 0
        except (ValueError, KeyboardInterrupt):
            return 0
    
    def list_sessions(self):
        """List available sessions."""
        sessions = list(self.sessions_dir.glob("*.session"))
        if not sessions:
            print("No saved sessions found.")
            return []
        
        print("\n📁 Available Sessions:")
        for i, session in enumerate(sessions, 1):
            session_name = session.stem
            size = session.stat().st_size
            mtime = session.stat().st_mtime
            import time
            date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            print(f"{i:2d}. {session_name} ({size} bytes, {date_str})")
        
        return sessions
    
    def list_transcripts(self):
        """List available transcripts."""
        transcripts = list(self.transcripts_dir.glob("*.txt"))
        if not transcripts:
            print("No transcripts found.")
            return []
        
        print("\n📝 Available Transcripts:")
        for i, transcript in enumerate(transcripts, 1):
            transcript_name = transcript.stem
            size = transcript.stat().st_size
            mtime = transcript.stat().st_mtime
            import time
            date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
            print(f"{i:2d}. {transcript_name} ({size} bytes, {date_str})")
        
        return transcripts
    
    def run_transcription(self, args):
        """Run the main transcription application."""
        print("\n🚀 Starting transcription system...")
        print("Note: The UI will appear in a moment. Press 'q' to quit when ready.")
        print("=" * 50)
        
        # Prepare the command
        micromamba_path = "/Users/max/.micromamba/bin/micromamba"
        cmd = f'eval "$({micromamba_path} shell hook --shell=bash)" && micromamba activate base && python main.py {args}'
        
        try:
            # Run the transcription system
            result = subprocess.run(cmd, shell=True, text=True)
            print(f"\nTranscription finished with exit code: {result.returncode}")
        except KeyboardInterrupt:
            print("\nTranscription interrupted by user.")
        except Exception as e:
            print(f"\nError running transcription: {e}")
        
        input("\nPress Enter to continue...")
    
    def start_microphone_transcription(self):
        """Start live microphone transcription."""
        print("\n🎙️ Live Microphone Transcription")
        print("-" * 40)
        
        session_name = input("Enter session name (or press Enter for auto-generated): ").strip()
        
        args = "--mic"
        if session_name:
            args += f" --session {session_name}"
        
        self.run_transcription(args)
    
    def transcribe_audio_file(self):
        """Transcribe an audio file."""
        print("\n📁 Audio File Transcription")
        print("-" * 40)
        
        audio_file = input("Enter path to audio file: ").strip()
        if not audio_file:
            print("No file specified.")
            input("Press Enter to continue...")
            return
        
        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            input("Press Enter to continue...")
            return
        
        session_name = input("Enter session name (or press Enter for auto-generated): ").strip()
        
        args = f"--input {audio_file}"
        if session_name:
            args += f" --session {session_name}"
        
        self.run_transcription(args)
    
    def resume_session(self):
        """Resume a previous session."""
        print("\n🔄 Resume Previous Session")
        print("-" * 40)
        
        sessions = self.list_sessions()
        if not sessions:
            input("Press Enter to continue...")
            return
        
        try:
            choice = int(input(f"\nSelect session (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                session_name = sessions[choice].stem
                args = f"--mic --resume --session {session_name}"
                self.run_transcription(args)
            else:
                print("Invalid selection.")
        except (ValueError, IndexError):
            print("Invalid selection.")
        
        input("Press Enter to continue...")
    
    def view_session_history(self):
        """View session history details."""
        print("\n📊 Session History")
        print("-" * 40)
        
        sessions = self.list_sessions()
        if not sessions:
            input("Press Enter to continue...")
            return
        
        try:
            choice = int(input(f"\nSelect session to view (1-{len(sessions)}): ")) - 1
            if 0 <= choice < len(sessions):
                session_file = sessions[choice]
                print(f"\n📄 Contents of {session_file.name}:")
                print("-" * 50)
                try:
                    with open(session_file, 'r') as f:
                        content = f.read()
                        if content.strip():
                            print(content)
                        else:
                            print("Session file is empty.")
                except Exception as e:
                    print(f"Error reading session: {e}")
            else:
                print("Invalid selection.")
        except (ValueError, IndexError):
            print("Invalid selection.")
        
        input("\nPress Enter to continue...")
    
    def view_transcripts(self):
        """View transcript details."""
        print("\n📝 View Transcripts")
        print("-" * 40)
        
        transcripts = self.list_transcripts()
        if not transcripts:
            input("Press Enter to continue...")
            return
        
        try:
            choice = int(input(f"\nSelect transcript to view (1-{len(transcripts)}): ")) - 1
            if 0 <= choice < len(transcripts):
                transcript_file = transcripts[choice]
                print(f"\n📄 Contents of {transcript_file.name}:")
                print("-" * 50)
                try:
                    with open(transcript_file, 'r') as f:
                        content = f.read()
                        if content.strip():
                            print(content)
                        else:
                            print("Transcript file is empty.")
                except Exception as e:
                    print(f"Error reading transcript: {e}")
            else:
                print("Invalid selection.")
        except (ValueError, IndexError):
            print("Invalid selection.")
        
        input("\nPress Enter to continue...")
    
    def clean_up_files(self):
        """Clean up old session and transcript files."""
        print("\n🧹 Clean Up Old Files")
        print("-" * 40)
        
        sessions = list(self.sessions_dir.glob("*.session"))
        transcripts = list(self.transcripts_dir.glob("*.txt"))
        
        print(f"Found {len(sessions)} session files and {len(transcripts)} transcript files.")
        
        if not sessions and not transcripts:
            print("No files to clean up.")
            input("Press Enter to continue...")
            return
        
        confirm = input("Delete all files? (y/N): ").strip().lower()
        if confirm == 'y':
            for session in sessions:
                session.unlink()
            for transcript in transcripts:
                transcript.unlink()
            print("All files deleted.")
        else:
            print("Cleanup cancelled.")
        
        input("Press Enter to continue...")
    
    def system_status(self):
        """Check system status and run tests."""
        print("\n⚙️ System Status & Test")
        print("-" * 40)
        
        # Check whisper.cpp
        whisper_cli = self.whisper_path / "build" / "bin" / "whisper-cli"
        if whisper_cli.exists():
            print("✅ whisper-cli binary found")
        else:
            print("❌ whisper-cli binary not found")
        
        # Check models
        tiny_model = self.whisper_path / "models" / "ggml-tiny.en.bin"
        base_model = self.whisper_path / "models" / "ggml-base.en.bin"
        
        if tiny_model.exists():
            size_mb = tiny_model.stat().st_size / (1024 * 1024)
            print(f"✅ Tiny model found ({size_mb:.1f} MB)")
        else:
            print("❌ Tiny model not found")
        
        if base_model.exists():
            size_mb = base_model.stat().st_size / (1024 * 1024)
            print(f"✅ Base model found ({size_mb:.1f} MB)")
        else:
            print("❌ Base model not found")
        
        # Check Python dependencies
        try:
            import pyaudio
            print("✅ PyAudio available (microphone support enabled)")
        except ImportError:
            print("❌ PyAudio not available (microphone support disabled)")
        
        try:
            import soundfile
            print("✅ SoundFile available")
        except ImportError:
            print("❌ SoundFile not available")
        
        try:
            import numpy
            print("✅ NumPy available")
        except ImportError:
            print("❌ NumPy not available")
        
        # Test whisper.cpp
        print("\nTesting whisper.cpp...")
        test_cmd = f'DYLD_LIBRARY_PATH={self.whisper_path}/build/src:{self.whisper_path}/build/ggml/src:{self.whisper_path}/build/ggml/src/ggml-metal:{self.whisper_path}/build/ggml/src/ggml-blas {whisper_cli} --help'
        
        try:
            result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ whisper.cpp test successful")
            else:
                print("❌ whisper.cpp test failed")
                print(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("❌ whisper.cpp test timed out")
        except Exception as e:
            print(f"❌ whisper.cpp test error: {e}")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main application loop."""
        while True:
            self.clear_screen()
            self.show_header()
            self.show_menu()
            
            choice = self.get_choice()
            
            if choice == 1:
                self.start_microphone_transcription()
            elif choice == 2:
                self.transcribe_audio_file()
            elif choice == 3:
                self.resume_session()
            elif choice == 4:
                self.view_session_history()
            elif choice == 5:
                self.view_transcripts()
            elif choice == 6:
                self.clean_up_files()
            elif choice == 7:
                self.system_status()
            elif choice == 8:
                print("\nGoodbye! 👋")
                break
            else:
                print("Invalid choice. Please select 1-8.")
                input("Press Enter to continue...")

if __name__ == "__main__":
    launcher = TranscriptionLauncher()
    try:
        launcher.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
