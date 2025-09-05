# Memory Optimization Fixes

## Problem Identified
The transcription system was experiencing severe memory issues:
- **30GB RAM usage** - Extremely high memory consumption
- **Memory pressure** - System becoming unresponsive
- **GUI freezing** - Live speech display stopping, scroll not updating
- **Process hanging** - Complete system freeze

## Root Causes Found

### 1. **Audio Buffer Accumulation**
- `audio_buffer` was growing indefinitely without cleanup
- No size limits on audio data storage
- Multiple copies of audio chunks created without deletion

### 2. **Queue Buildup**
- Audio and refine queues had no size limits
- Queues could grow indefinitely during processing delays
- No queue overflow handling

### 3. **Memory Leaks in Threading**
- Audio chunks copied multiple times without cleanup
- No explicit memory deallocation
- Garbage collection not forced

### 4. **Large Model Memory**
- Two large models loaded simultaneously (Distil-Whisper + Whisper-Large-v3)
- No model memory cleanup
- CUDA cache not cleared

## Fixes Implemented

### 1. **Enhanced Memory Management**
```python
# Added memory monitoring
import psutil
import gc

# Queue size limits
self.audio_queue = queue.Queue(maxsize=5)  # Limit to 5 chunks
self.refine_queue = queue.Queue(maxsize=3)  # Limit to 3 chunks

# Buffer size limits
self.max_buffer_size = self.chunk_samples * 2  # Keep only 2 chunks worth
```

### 2. **Aggressive Cleanup**
```python
def _cleanup_memory(self):
    # Clear audio buffer
    self.audio_buffer = np.array([], dtype=np.float32)
    
    # Clear queues
    while not self.audio_queue.empty():
        self.audio_queue.get_nowait()
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 3. **Memory Monitoring**
```python
def _check_memory_usage(self):
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 1000:  # Warning at 1GB
        print(f"⚠️  High memory usage: {memory_mb:.1f} MB")
    if memory_mb > 2000:  # Force cleanup at 2GB
        self._cleanup_memory()
```

### 4. **Queue Overflow Protection**
```python
# Try to put in queue, but don't block if full
try:
    self.audio_queue.put_nowait(chunk)
except queue.Full:
    print("⚠️  Audio queue full, skipping chunk")
    del chunk  # Explicitly delete to free memory
```

### 5. **Explicit Memory Deallocation**
```python
# Explicitly delete audio chunk to free memory
del audio_chunk
del job
```

## Lightweight Alternative

Created `engine_lightweight.py` and `gui_lightweight.py` for extreme memory optimization:

### Features:
- **Single Model**: Uses `whisper-tiny.en` instead of two large models
- **Shorter Chunks**: 5-second chunks instead of 15-second
- **Smaller Queues**: Max 3 chunks instead of unlimited
- **Aggressive Cleanup**: Every 10 seconds instead of 30
- **Single Pass**: No two-pass system to reduce complexity

### Usage:
```bash
./start.sh --lightweight
# or
./start.sh -l
```

## Memory Monitoring Tools

### 1. **Built-in Monitoring**
- Real-time memory usage display in GUI
- Automatic cleanup when memory exceeds thresholds
- Periodic memory checks every 10-30 seconds

### 2. **Standalone Monitor**
```bash
python memory_monitor.py [PID]
```
- Monitors specific process memory usage
- Auto-detects transcription processes
- Real-time memory tracking

## Expected Results

### Full System (Fixed)
- **Memory Usage**: Should stay under 2GB during normal operation
- **Stability**: No more hanging or freezing
- **Performance**: Smooth GUI updates and scrolling

### Lightweight System
- **Memory Usage**: Should stay under 500MB
- **Speed**: Faster startup and processing
- **Reliability**: More stable for long sessions

## Testing Recommendations

1. **Start with Lightweight**: Use `./start.sh --lightweight` first
2. **Monitor Memory**: Run `python memory_monitor.py` in another terminal
3. **Test Duration**: Run for 30+ minutes to test stability
4. **Check GUI**: Ensure live speech updates and history scrolls properly

## Files Modified

- `engine.py` - Added memory management and monitoring
- `requirements.txt` - Added psutil for memory monitoring
- `start.sh` - Added lightweight mode option
- `engine_lightweight.py` - New lightweight engine
- `gui_lightweight.py` - New lightweight GUI
- `memory_monitor.py` - Standalone memory monitoring tool

The system should now be much more stable and use significantly less memory!
