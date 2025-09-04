# Lecture Transcription System

A robust two-pass transcription system for converting lecture audio into clean, accurate text optimized for AI consumption.

## Current Status ✅

**System is fully operational** with the following features:

### Two-Pass Transcription System

- **Streaming Pass**: Real-time transcription using `distil-whisper/distil-medium.en` (fast, English-only)
- **Refining Pass**: High-accuracy transcription using `openai/whisper-large-v3` (best quality)
- **Sentence-Aware Processing**: 15-second chunks with intelligent sentence boundary detection
- **English-Only Mode**: Optimized for English lectures with no language detection overhead

### Live GUI Interface

- **Real-time Display**: Shows current speech in "Live Speech" section
- **Scrollable History**: Accumulates finalized transcripts in "Finalized Transcript" section
- **Clean Interface**: Monospace font, dark theme, Apple Music-style layout
- **Automatic Saving**: Sessions saved to `transcripts/` folder as TXT files

### Session Management

- **Auto-Save**: Transcripts automatically saved on quit (including force quit with Ctrl+C)
- **Signal Handling**: Graceful shutdown with session preservation
- **TXT Output**: Simple, clean text format optimized for AI consumption
- **Timestamped Files**: Each session gets a unique filename with timestamp

### Full Audio Transcription

- **Post-Session Processing**: `full_transcribe.py` for transcribing complete audio files
- **Best Model**: Uses `openai/whisper-large-v3` for maximum accuracy
- **Batch Processing**: Command-line tool for processing recorded lectures

## Architecture

```
Microphone → Audio Buffer (15s chunks) → Streaming Model (Distil-Whisper) → Sentence Detection
                ↓                              ↓                              ↓
         Real-time capture            Fast transcription              Complete sentences
                                                                    ↓
         Refining Model (Whisper-Large-v3) ← Audio Queue ← Sentence Buffer
                ↓
         High-accuracy transcription → GUI Display → TXT File Save
```

## Installation

### Prerequisites

1. **Python Environment**: Uses `micromamba` for dependency management
2. **Audio Support**: `pyaudio` for microphone input
3. **AI Models**: Hugging Face Transformers for local ASR

### Setup

1. **Activate Environment**:
   ```bash
   ./start.sh
   ```

2. **Dependencies**: Automatically installed in isolated `lecture-transcriber` environment:
   - `transformers>=4.42.0` - Hugging Face ASR models
   - `torch>=2.3.0` - PyTorch with MPS support for Apple Silicon
   - `accelerate>=0.31.0` - Model acceleration
   - `pyaudio` - Microphone input
   - `librosa` - Audio processing
   - `soundfile` - Audio file handling

### Model Download

The system automatically downloads required models:

- `distil-whisper/distil-medium.en` (~789MB) - For streaming/fast transcription
- `openai/whisper-large-v3` (~3.09GB) - For refining/accurate transcription

## Usage

### Live Transcription (Primary Mode)

```bash
# Start the GUI application
./start.sh
```

This launches the main transcription interface with:
- Real-time microphone input
- Live speech display
- Scrollable transcript history
- Automatic session saving

### Post-Session Full Transcription

```bash
# Transcribe a complete audio file with maximum accuracy
python full_transcribe.py lecture.wav

# Specify custom output file
python full_transcribe.py lecture.wav my_transcript.txt
```

### Session Files

All sessions are automatically saved to the `transcripts/` folder:
- Format: `session_YYYYMMDD_HHMMSS.txt`
- Contains: All refined transcriptions with timestamps
- Auto-saved: On quit, force quit (Ctrl+C), or crash

## Performance Characteristics

### Streaming Pass (Distil-Whisper)

- **Speed**: ~3-5x faster than large models
- **Accuracy**: Good for real-time display
- **Memory**: ~789MB model size
- **Use Case**: Live transcription, sentence detection

### Refining Pass (Whisper-Large-v3)

- **Speed**: Standard transcription speed
- **Accuracy**: Highest available accuracy
- **Memory**: ~3.09GB model size
- **Use Case**: Final transcripts, archival quality

## File Structure

```
lecture_transcriber/
├── gui_transcriber.py           # Main GUI application
├── engine.py                    # Core transcription engine
├── full_transcribe.py           # Post-session transcription tool
├── start.sh                     # Environment launcher
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── project_goals.md             # Project objectives
└── transcripts/                 # Session output folder
    └── session_*.txt            # Auto-saved session files
```

## Integration with AI Systems

The plain text output is optimized for AI consumption:

```python
# Load session transcript for AI processing
with open('transcripts/session_20250904_141451.txt', 'r') as f:
    transcript = f.read()

# Use with local LLM or cloud AI
response = ai_system.query(f"Summarize this lecture: {transcript}")
```

### Batch Processing

```bash
# Process multiple audio files
for file in *.wav; do
  python full_transcribe.py "$file"
done
```

## Troubleshooting

### Common Issues

1. **"Cannot specify task or language for English-only model"**
   - This is normal - English-only models don't need language parameters
   - The system automatically handles this

2. **"Model not found"**
   - Models are downloaded automatically on first run
   - Check internet connection and disk space (4GB+ needed)

3. **"Microphone not working"**
   - Check system microphone permissions
   - Ensure no other apps are using the microphone

4. **"Session not saved"**
   - Sessions auto-save on quit (including Ctrl+C)
   - Check `transcripts/` folder for saved files

### Performance Tips

- Use SSD storage for faster model loading
- Ensure sufficient RAM (8GB+ recommended for large models)
- Close other applications during transcription
- Apple Silicon Macs get best performance with MPS acceleration

## Roadmap 🗺️

### Phase 1: Core System ✅ COMPLETED
- [x] Two-pass transcription system
- [x] Live GUI interface
- [x] Session management and auto-save
- [x] English-only optimization
- [x] Sentence-aware processing
- [x] Signal handling for graceful shutdown

### Phase 2: RAG Integration (Next Priority)
- [ ] **Vector Database Setup**: Implement local vector storage for transcript embeddings
- [ ] **Embedding Pipeline**: Generate embeddings for all session transcripts
- [ ] **Q&A Interface**: Add question-answering capability to the GUI
- [ ] **Context Retrieval**: Smart retrieval of relevant transcript segments
- [ ] **Local LLM Integration**: Connect with local language models for responses

### Phase 3: Advanced Features
- [ ] **Speaker Diarization**: Identify and separate different speakers
- [ ] **Topic Segmentation**: Automatic detection of lecture topics/sections
- [ ] **Keyword Extraction**: Identify key terms and concepts
- [ ] **Summary Generation**: Automatic lecture summaries
- [ ] **Search Interface**: Full-text search across all transcripts

### Phase 4: Enhanced User Experience
- [ ] **Web Interface**: Browser-based transcription interface
- [ ] **Mobile App**: iOS/Android companion app
- [ ] **Cloud Sync**: Optional cloud storage and sync
- [ ] **Collaboration**: Multi-user session sharing
- [ ] **Export Options**: PDF, Word, and other format exports

### Phase 5: AI-Powered Features
- [ ] **Lecture Analysis**: Automatic analysis of lecture structure and content
- [ ] **Study Guide Generation**: Create study materials from transcripts
- [ ] **Quiz Generation**: Automatic quiz creation from lecture content
- [ ] **Note Integration**: Connect with existing note-taking apps
- [ ] **Real-time Translation**: Multi-language support with translation

## Contributing

This system is designed for educational use and AI research. Contributions are welcome for:

- Performance optimizations
- Additional output formats
- Integration improvements
- Documentation enhancements

## License

This project uses whisper.cpp under the MIT License. See `whisper.cpp/LICENSE` for details.
