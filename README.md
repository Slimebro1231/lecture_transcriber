# Lecture Transcription System

A robust two-pass transcription system for converting lecture audio into clean, accurate text optimized for AI consumption.

## Features

### Two-Pass Transcription

- **Streaming Pass**: Fast, real-time transcription using a lightweight model
- **Refining Pass**: High-accuracy transcription using a larger, more powerful model
- **Context Integration**: Second pass uses first pass results for improved accuracy

### Resumable Transcription

- Continue transcription from specific timestamps
- Resume from partial transcripts
- Handle interrupted sessions gracefully

### Live Streaming

- Real-time transcription during lectures
- Microphone input support
- Live scrolling transcript display
- Configurable update intervals

### Multiple Output Formats

- Plain text (.txt) - Optimized for AI consumption
- Markdown (.md) - Structured format with headers
- JSON (.json) - Machine-readable with metadata

## Architecture

```
Audio Input → Streaming Pass (tiny model) → Refining Pass (base model) → Final Output
                ↓                              ↓
         Fast transcription            High-accuracy transcription
         Real-time display             Context-aware refinement
```

## Installation

### Prerequisites

1. **whisper.cpp** (already installed in your workspace)
2. **ffmpeg** for audio processing:

   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```
3. **Python Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Model Download

The system automatically downloads required models:

- `ggml-tiny.en.bin` (39MB) - For streaming/fast transcription
- `ggml-base.en.bin` (141MB) - For refining/accurate transcription

## Usage

### Basic Two-Pass Transcription

```bash
# Transcribe an audio file with both passes
python two_pass_transcriber.py --input lecture.wav --output transcript.txt

# Specify output format
python two_pass_transcriber.py --input lecture.wav --output transcript.md --format md
```

### Streaming-Only Mode

```bash
# Fast transcription only (streaming pass)
python two_pass_transcriber.py --stream --input lecture.wav --output fast_transcript.txt
```

### Refining-Only Mode

```bash
# High-accuracy transcription only (refining pass)
python two_pass_transcriber.py --refine --input lecture.wav --output accurate_transcript.txt
```

### Resume Transcription

```bash
# Continue from a specific timestamp
python two_pass_transcriber.py --resume --input lecture.wav --partial partial.txt --start-time 300.0

# Resume from beginning with partial transcript
python two_pass_transcriber.py --resume --input lecture.wav --partial partial.txt
```

### Live Streaming

```bash
# Live transcription from microphone
python live_streaming_transcriber.py --mic

# Live transcription from audio file (simulated real-time)
python live_streaming_transcriber.py --input lecture.wav

# Save live transcript
python live_streaming_transcriber.py --input lecture.wav --output live_transcript.txt
```

## Command Line Options

### Two-Pass Transcriber

| Option             | Description                        | Default        |
| ------------------ | ---------------------------------- | -------------- |
| `--input, -i`    | Input audio file                   | Required       |
| `--output, -o`   | Output transcript file             | Auto-generated |
| `--stream`       | Streaming mode only                | False          |
| `--refine`       | Refining mode only                 | False          |
| `--resume`       | Resume from partial transcript     | False          |
| `--partial`      | Partial transcript file for resume | None           |
| `--start-time`   | Start time in seconds for resume   | 0.0            |
| `--format`       | Output format (txt, md, json)      | txt            |
| `--whisper-path` | Path to whisper.cpp                | ./whisper.cpp  |

### Live Streaming Transcriber

| Option               | Description                    | Default          |
| -------------------- | ------------------------------ | ---------------- |
| `--input, -i`      | Input audio file               | None             |
| `--mic`            | Use microphone input           | False            |
| `--output, -o`     | Output transcript file         | None             |
| `--model, -m`      | Whisper model to use           | ggml-tiny.en.bin |
| `--chunk-duration` | Audio chunk duration (seconds) | 3.0              |
| `--whisper-path`   | Path to whisper.cpp            | ./whisper.cpp    |

## Performance Characteristics

### Streaming Pass (tiny model)

- **Speed**: ~3-5x faster than base model
- **Accuracy**: Good for real-time display
- **Memory**: ~39MB model size
- **Use Case**: Live transcription, quick previews

### Refining Pass (base model)

- **Speed**: Standard transcription speed
- **Accuracy**: High accuracy with context
- **Memory**: ~141MB model size
- **Use Case**: Final transcripts, archival

## Testing

Run the comprehensive test suite:

```bash
python test_transcription_system.py
```

This will verify:

- whisper.cpp installation
- Model availability
- Audio file processing
- ffmpeg availability
- Python dependencies
- Transcription functionality

## File Structure

```
lecture_transcriber/
├── two_pass_transcriber.py      # Main two-pass system
├── live_streaming_transcriber.py # Real-time transcription
├── test_transcription_system.py  # Comprehensive testing
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── project_goals.md             # Project objectives
└── whisper.cpp/                 # Core transcription engine
    ├── build/bin/main           # Whisper binary
    └── models/                  # Model files
        ├── ggml-tiny.en.bin     # Streaming model
        └── ggml-base.en.bin     # Refining model
```

## Advanced Usage

### Custom Model Paths

```bash
python two_pass_transcriber.py \
  --input lecture.wav \
  --whisper-path /path/to/whisper.cpp \
  --output transcript.txt
```

### Batch Processing

```bash
# Process multiple files
for file in *.wav; do
  python two_pass_transcriber.py --input "$file" --output "${file%.wav}_transcript.txt"
done
```

### Integration with AI Systems

The plain text output is optimized for AI consumption:

```python
# Load transcript for AI processing
with open('transcript.txt', 'r') as f:
    transcript = f.read()

# Use with local LLM or cloud AI
response = ai_system.query(f"Summarize this lecture: {transcript}")
```

## Troubleshooting

### Common Issues

1. **"whisper.cpp not found"**

   - Ensure whisper.cpp is in the current directory
   - Use `--whisper-path` to specify custom location
2. **"Model not found"**

   - Models are downloaded automatically
   - Check internet connection and disk space
3. **"ffmpeg not found"**

   - Install ffmpeg system-wide
   - Ensure it's in your PATH
4. **Audio format issues**

   - Convert audio to WAV format
   - Ensure 16kHz sample rate for best results

### Performance Tips

- Use SSD storage for faster model loading
- Ensure sufficient RAM (2GB+ recommended)
- Close other applications during transcription
- Use appropriate chunk sizes for live streaming

## Future Enhancements

- [ ] RAG integration for Q&A
- [ ] Speaker diarization
- [ ] Multi-language support
- [ ] Cloud model integration
- [ ] Web interface
- [ ] Mobile app support

## Contributing

This system is designed for educational use and AI research. Contributions are welcome for:

- Performance optimizations
- Additional output formats
- Integration improvements
- Documentation enhancements

## License

This project uses whisper.cpp under the MIT License. See `whisper.cpp/LICENSE` for details.
