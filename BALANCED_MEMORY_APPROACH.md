# Balanced Memory Management Approach

## User Feedback
- 30GB should be plenty for audio AI
- Memory cleanup may be too sensitive (AI models are already 3GB)
- As long as system doesn't hang, it's fine
- Memory cleanup may split up sentence structure

## Adjusted Approach

### 1. **Realistic Memory Thresholds**
```python
# OLD (too aggressive)
if memory_mb > 1000:  # Warning at 1GB
if memory_mb > 2000:  # Force cleanup at 2GB

# NEW (realistic for 30GB system)
if memory_mb > 8000:  # Warning at 8GB (models + processing)
if memory_mb > 15000: # Force cleanup at 15GB (half of available)
```

### 2. **Preserve Sentence Structure**
```python
# OLD (breaks sentences)
self.sentence_buffer = ""  # Cleared completely

# NEW (preserves sentences)
# self.sentence_buffer = ""  # Commented out to preserve sentences
```

### 3. **Less Aggressive Buffer Management**
```python
# OLD (too restrictive)
self.max_buffer_size = self.chunk_samples * 2  # Only 2 chunks

# NEW (more reasonable)
self.max_buffer_size = self.chunk_samples * 10  # Keep 10 chunks worth
```

### 4. **Smarter Queue Management**
```python
# OLD (immediate skip)
except queue.Full:
    print("⚠️  Audio queue full, skipping chunk")
    del chunk

# NEW (retry before skipping)
except queue.Full:
    time.sleep(0.1)  # Wait a bit
    try:
        self.audio_queue.put_nowait(chunk)
    except queue.Full:
        print("⚠️  Audio queue persistently full, skipping chunk")
        del chunk
```

### 5. **Selective Cleanup**
```python
# Only clear excess, preserve recent data
if len(self.audio_buffer) > self.max_buffer_size:
    excess = len(self.audio_buffer) - self.max_buffer_size
    self.audio_buffer = self.audio_buffer[excess:]  # Keep recent data

# Keep recent queue items
if audio_queue_size > 3:  # Keep 3 most recent
    for _ in range(audio_queue_size - 3):
        self.audio_queue.get_nowait()  # Clear only old items
```

## Expected Behavior

### Memory Usage
- **Normal Operation**: 3-8GB (models + processing)
- **Warning Threshold**: 8GB (just informational)
- **Cleanup Threshold**: 15GB (half of 30GB available)
- **No Hanging**: System should remain responsive

### Sentence Processing
- **Preserved Structure**: Sentence buffer never cleared
- **Continuous Flow**: No interruption of sentence building
- **Quality Maintained**: Two-pass system works as intended

### Performance
- **Smooth Operation**: No aggressive cleanup interrupting processing
- **Responsive GUI**: Live speech updates and scrollable history
- **Stable Long Sessions**: Can run for hours without issues

## Key Changes Made

1. **Increased buffer size** from 2 to 10 chunks
2. **Raised memory thresholds** from 1-2GB to 8-15GB
3. **Preserved sentence buffer** during cleanup
4. **Added retry logic** for queue operations
5. **Selective cleanup** that keeps recent data
6. **Less frequent checks** (60 seconds instead of 30)

## Testing Recommendation

The system should now:
- Use reasonable amounts of memory (3-8GB normal, up to 15GB before cleanup)
- Maintain sentence structure and quality
- Not hang or freeze
- Provide smooth GUI experience
- Handle long transcription sessions well

This balanced approach respects your 30GB system while still preventing the hanging issues that occurred before.
